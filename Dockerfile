# Vichara container. Also the Hugging Face Space image.
#
# Python and Node in one image, because the default sandbox backend is Pyodide
# hosted by Node. A Python-only image would start, pass its health check, and
# silently report `run_python` as unavailable -- the capability set would shrink
# and nobody would notice until an eval task needed arithmetic.
#
# Single stage on purpose. A builder stage would have to copy node_modules and
# the uv venv across, and both are large enough that the copy costs more than
# the layers it saves. Simplicity wins here.

FROM python:3.11-slim

# Node from nodesource: Debian's own node is too old for the Pyodide package.
# --no-install-recommends and the apt-list cleanup in the same layer keep this
# from adding a few hundred megabytes of documentation nobody reads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove gnupg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Hugging Face Spaces runs the container as uid 1000 and mounts nothing
# writable outside /home/user. Everything the agent writes -- session
# workspaces, the response cache, checkpoints, trajectories -- has to live
# under a path this user owns, or the first run fails on a permission error
# that reads like a config bug.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv

WORKDIR /home/user/app

# Dependencies before source, so a code change does not re-resolve the
# environment or re-download Pyodide. This is the layer that takes minutes.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user package.json package-lock.json ./
RUN npm ci --omit=dev

COPY --chown=user . .
RUN uv sync --frozen --no-dev

# Gradio's default. Spaces expects 7860 and ignores EXPOSE, but a local
# `docker run -p 7860:7860` needs it.
EXPOSE 7860

ENV WORKSPACE_ROOT=/home/user/app/sessions \
    CACHE_PATH=/home/user/app/data/cache/llm_cache.sqlite \
    CHECKPOINT_PATH=/home/user/app/data/cache/checkpoints.sqlite \
    LOG_FORMAT=json \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# The same command a developer runs. It exits non-zero on a broken
# configuration and zero on a merely degraded one, which is exactly the
# distinction a container healthcheck wants.
HEALTHCHECK --interval=60s --timeout=20s --start-period=30s --retries=3 \
    CMD uv run vichara health || exit 1

CMD ["uv", "run", "python", "app.py"]
