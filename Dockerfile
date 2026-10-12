# Every PoseYdon image, as named stages of one file.
#
# The split of responsibilities: this file provides the OS and the system
# packages; every Python dependency is declared in pyproject.toml and pinned in
# uv.lock, and each image installs it with one `uv sync --locked`. So no image
# carries a Python package the lock does not name, and `test`, `train` and
# `compat` cannot drift apart: they are the same locked set, differing only in
# the extras each selects. The one exception is `fbx`, whose Python is
# Blender's -- see that stage.
#
# Adding or upgrading a Python dependency is `uv add <pkg> [--optional <extra>]`
# (or `uv lock --upgrade-package <pkg>`) on the host, then
# `docker compose build <service>`.
#
# Build one with `docker compose build <service>`; each service names its
# `target:` in docker-compose.yml. BuildKit builds only the stage asked for and
# its ancestors, so `fbx` never pulls the CUDA wheels and `test` never pulls
# Blender.

# ---------------------------------------------------------------------------
# base -- the OS, uv, and the project's dependency manifest.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base

# uv pinned to the version that writes uv.lock, so the lock format always
# matches the reader. Copied in rather than taken as the base image because
# uv's own 0.13 images ship only on Debian trixie, and the OS stays bookworm.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

WORKDIR /app

# UV_PROJECT_ENVIRONMENT puts the virtualenv OUTSIDE /app so the bind mount in
# docker-compose.yml cannot shadow it. UV_LINK_MODE=copy avoids hardlink warnings
# when the uv cache and the venv live on different layers.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Only the manifest and the lock are copied, so editing source never
# invalidates the dependency layer. --no-install-project means the package
# itself is not built here; pytest's `pythonpath = ["src"]` setting makes it
# importable from the bind mount instead.
COPY pyproject.toml uv.lock ./

# ---------------------------------------------------------------------------
# test -- the default. CPU only; nothing in the suite needs a GPU.
# ---------------------------------------------------------------------------
FROM base AS test

RUN uv sync --locked --no-install-project --extra dev --extra cpu

CMD ["pytest", "-v"]

# ---------------------------------------------------------------------------
# train -- CUDA 13 aarch64/sbsa, for the GB10 (Blackwell).
# ---------------------------------------------------------------------------
FROM base AS train

# `cu130` selects torch from PyTorch's CUDA 13.0 index instead of the CPU one;
# the two extras conflict, so exactly one torch is ever installed.
#
# Route chosen on measurement, not preference (design spec §4, "The aarch64
# image"): this works end to end on the GB10 and an NGC PyTorch aarch64 base
# was deliberately not built, because a ~20 GB pull to compare a route we do
# not need is not evidence worth buying.
#
# NOT installed: build-essential. It is needed only by torch.compile, which
# this run does not use -- measured at +8% on a median batch and 2.3x WORSE on
# the largest rig, because dynamic joint counts recompile constantly.
RUN uv sync --locked --no-install-project --extra dev --extra cu130 --extra train

CMD ["python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"]

# ---------------------------------------------------------------------------
# compat -- reference comparison. Never a dependency of the suite.
# ---------------------------------------------------------------------------
# Kept out of `test` so the main suite never depends on `Motion`, on network
# access, or on a 900MB T5 download. Used to verify PoseYdon against the
# published implementation: reading its cond.npy, building its T5 joint-name
# embeddings, running its models, and benchmarking its BVH loader.
FROM base AS compat

# git fetches the reference package (pinned by commit in uv.lock); a compiler
# builds it; ffmpeg is for the reference's video writer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app/src:/app

# The `compat` extra carries the reference (`motion`) and what its scripts
# import. Its build needs a setuptools old enough to ship pkg_resources (< 81);
# that pin is declared in pyproject's [tool.uv.extra-build-dependencies], so the
# build stays isolated and nothing here installs packages by hand.
RUN uv sync --locked --no-install-project --extra dev --extra cpu --extra compat

CMD ["python", "-c", "import Animation, BVH, Quaternions; print('Motion available')"]

# ---------------------------------------------------------------------------
# fbx -- Blender, headless, for FBX preprocessing.
# ---------------------------------------------------------------------------
# The one stage that shares nothing with `base`, and cannot: Blender links
# against the SYSTEM Python, so it needs distro packages rather than a venv.
# This is the deliberate exception to "every Python dependency comes from
# uv.lock": here Python packages ARE system dependencies, because they must be
# built against the interpreter Blender embeds.
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
# libegl1 + the Mesa drivers, for EEVEE. Blender's rasterizing engines --
# EEVEE and Workbench -- need an OpenGL context, and a headless container has
# no display to get one from: without these, ANY render aborts the process
# outright with `Couldn't open libEGL.so.1` before a single frame is drawn.
# libegl1 supplies the EGL loader that lets Blender create a context with no X
# server or Wayland compositor, and libgl1-mesa-dri supplies llvmpipe, Mesa's
# software rasterizer, which is what actually executes the GL calls on a
# machine whose GPU is not attached to this container (the `fbx` service takes
# no CDI device, deliberately -- see docker-compose.yml).
#
# This is not a preference between renderers. Cycles is a CPU path tracer and
# needs no GL at all, so it was the only engine that ran here before -- but it
# path-traces every frame across every core, which is a wholly unreasonable
# price for a preview whose entire job is to confirm a mesh follows its
# skeleton. EEVEE rasterizes, and on llvmpipe it is far cheaper.
RUN apt-get update && apt-get install -y --no-install-recommends \
      blender python3-yaml python3-numpy python3-scipy python3-pytest \
      python3-matplotlib python3-imageio \
      libegl1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
