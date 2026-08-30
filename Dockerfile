# Test and development environment for PoseYdon.
# The host needs only Docker: no Python, pip or uv installation required.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

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

CMD ["pytest", "-v"]
