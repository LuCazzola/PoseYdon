# Every PoseYdon image, as named stages of one file.
#
# There used to be four Dockerfiles. Three of them shared a base AND a
# dependency layer -- `Dockerfile.train` repeated `Dockerfile`'s
# `uv sync --extra dev --no-install-project` verbatim -- which meant the image
# training runs in could silently drift from the image the suite is tested in.
# One `base` stage removes that class of bug: `test`, `train` and `compat` are
# provably the same dependency set plus their own additions.
#
# Build one with `docker compose build <service>`; each service names its
# `target:` in docker-compose.yml. BuildKit builds only the stage asked for and
# its ancestors, so `fbx` never pulls the CUDA wheels and `test` never pulls
# Blender.

# ---------------------------------------------------------------------------
# base -- the dependency set the suite is tested against.
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

# UV_PROJECT_ENVIRONMENT puts the virtualenv OUTSIDE /app so the bind mount in
# docker-compose.yml cannot shadow it. UV_LINK_MODE=copy avoids hardlink warnings
# when the uv cache and the venv live on different layers.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Only pyproject.toml is copied, so editing source never invalidates this layer.
# --no-install-project means the package itself is not built here; pytest's
# `pythonpath = ["src"]` setting makes it importable from the bind mount instead.
COPY pyproject.toml ./
RUN uv sync --extra dev --no-install-project

# ---------------------------------------------------------------------------
# test -- the default. CPU only; nothing in the suite needs a GPU.
# ---------------------------------------------------------------------------
FROM base AS test

CMD ["pytest", "-v"]

# ---------------------------------------------------------------------------
# train -- CUDA 13 aarch64/sbsa, for the GB10 (Blackwell).
# ---------------------------------------------------------------------------
FROM base AS train

# `base` honours pyproject's [tool.uv.sources] pin and installs CPU torch; this
# replaces exactly that one package with the CUDA build. Done this way round so
# the shared pyproject.toml -- and therefore the CPU test image -- needs no
# change at all.
#
# Route chosen on measurement, not preference (design spec §4, "The aarch64
# image"): this works end to end on the GB10 and an NGC PyTorch aarch64 base
# was deliberately not built, because a ~20 GB pull to compare a route we do
# not need is not evidence worth buying.
#
# NOT installed: build-essential. It is needed only by torch.compile, which
# this run does not use -- measured at +8% on a median batch and 2.3x WORSE on
# the largest rig, because dynamic joint counts recompile constantly.
RUN uv pip install --python /opt/venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu130 \
      --reinstall-package torch torch && \
    uv pip install --python /opt/venv/bin/python wandb

CMD ["python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"]

# ---------------------------------------------------------------------------
# compat -- reference comparison. Never a dependency of the suite.
# ---------------------------------------------------------------------------
# Kept out of `test` so the main suite never depends on `Motion`, on network
# access, or on a 900MB T5 download. Used to verify PoseYdon against the
# published implementation: reading its cond.npy, building its T5 joint-name
# embeddings, running its models, and benchmarking its BVH loader.
FROM base AS compat

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app/src:/app

# The reference's own dependency, which is why its cond.npy cannot be opened
# without it. Installed only here.
# Motion declares no build-system requirements, so --no-build-isolation needs
# setuptools already present -- and specifically a setuptools old enough to still
# ship pkg_resources, removed in 81. Two pins to install one package: part of why
# PoseYdon vendors its own BVH IO instead.
RUN uv pip install "setuptools<81" wheel \
    && uv pip install --no-build-isolation "git+https://github.com/inbar-2344/Motion.git" \
    && uv pip install transformers sentencepiece num2words matplotlib imageio imageio-ffmpeg "moviepy<2" einops blobfile

CMD ["python", "-c", "import Animation, BVH, Quaternions; print('Motion available')"]

# ---------------------------------------------------------------------------
# fbx -- Blender, headless, for FBX preprocessing.
# ---------------------------------------------------------------------------
# The one stage that shares nothing with `base`, and cannot: Blender links
# against the SYSTEM Python, so it needs distro packages rather than a venv.
# Blender's Python API is the only mature FBX reader available here -- and
# `pip install bpy` is not an option on arm64, which has no bpy wheels on PyPI,
# so this uses the distro Blender driven with `blender --background --python`.
FROM debian:bookworm-slim AS fbx

# Blender links against the system Python, so manifest parsing needs the
# distro's PyYAML rather than a pip install into a bundled interpreter.
# python3-numpy and python3-scipy are needed transitively, and scipy in
# particular is NOT dead weight to "optimise away": the FBX script imports
# poseydon.ingest.pipeline.available_rigs (to share the entity-first layout
# with the BVH script), which imports poseydon.core.animation, which imports
# poseydon.core.kinematics, which does `from poseydon.core.rotations import
# quat_apply, quat_mul` -- and poseydon.core.rotations does `from
# scipy.spatial.transform import Rotation` at MODULE SCOPE. Nothing the FBX
# script calls actually touches Rotation (available_rigs is a plain directory
# scan), but Python executes a module's whole body on import, so this whole
# chain -- and therefore the FBX script itself -- fails to import at all
# without scipy installed (verified: removing it reproduces
# `ModuleNotFoundError: No module named 'scipy'` at the available_rigs import
# line, before the script's own code ever runs). FBX's own mesh.npz writer
# uses numpy directly too.
#
# python3-pytest, for the same "links against the system Python" reason: this
# image has no pip and no venv, so `pip install pytest` is not an option --
# and it is not optional dead weight either. `tests/build/test_roundtrip_fbx.py`
# is a real pytest test that is gated on Blender (`pytest.importorskip("bpy",
# ...)`), meaning it can ONLY ever execute inside this image. Without a pytest
# runner in here, that test skips forever in `test` (no Blender) and cannot
# even be collected in here (no pytest) -- a permanently-skipping test that
# reports green either way, which is exactly the failure mode this whole plan
# was written to close (`test_round_trip_returns_the_source_rig[Crab]` skipped
# silently for weeks looking like a pass). Debian ships a distro
# `python3-pytest` package built against the system interpreter, so it is
# importable from inside Blender the same way PyYAML/numpy/scipy are.
RUN apt-get update && apt-get install -y --no-install-recommends \
      blender python3-yaml python3-numpy python3-scipy python3-pytest \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
