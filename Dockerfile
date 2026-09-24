# Simple single-stage GPU image that runs `tide2-runner`.
# hatch-vcs falls back to pyproject's fallback-version, so no .git is needed here.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# uv package manager (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/tide2/.venv \
    PATH="/opt/tide2/.venv/bin:$PATH" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN uv python install 3.12

WORKDIR /opt/tide2

# Dependency layer (cached): lockfiles + the metadata files pyproject references.
COPY pyproject.toml uv.lock README.md LICENSE-MIT ./
RUN uv sync --locked --no-install-project --no-dev

# Project source, then install the project itself.
COPY src ./src
RUN uv sync --locked --no-dev

ENTRYPOINT ["tide2-runner"]
