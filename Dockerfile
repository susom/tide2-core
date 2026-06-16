ARG GPU_BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
ARG CPU_BASE_IMAGE=ubuntu:24.04
ARG USER=tide
ARG GROUP=tide
ARG USER_UID=5000
ARG USER_GID=5000
FROM ${GPU_BASE_IMAGE} AS base

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

RUN echo UTC > /etc/timezone

ARG USER
ARG GROUP
ARG USER_UID
ARG USER_GID
ARG USER_SHELL=/usr/sbin/nologin
RUN groupadd -f -g ${USER_GID} ${GROUP} && \
    useradd -u ${USER_UID} \
    -d /home/${USER} \
    -s ${USER_SHELL} \
    -g ${USER_GID} \
    -m \
    ${USER}

# Set up Python with uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && \
    chown -R ${USER}:${GROUP} /opt/python

WORKDIR /opt/tide2

RUN chown ${USER}:${GROUP} /opt/tide2 && \
    chmod 755 /opt/tide2

ENV TIDE_CACHE_DIR=/home/${USER}/.cache/tide
RUN mkdir -p ${TIDE_CACHE_DIR} && \
    chown -R ${USER}:${GROUP} /home/${USER}

# ============================================================================
# CPU-only base (no CUDA libraries)
# ============================================================================
FROM ${CPU_BASE_IMAGE} AS cpu-base

COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

RUN echo UTC > /etc/timezone

ARG USER
ARG GROUP
ARG USER_UID
ARG USER_GID
ARG USER_SHELL=/usr/sbin/nologin
RUN groupadd -f -g ${USER_GID} ${GROUP} && \
    useradd -u ${USER_UID} \
    -d /home/${USER} \
    -s ${USER_SHELL} \
    -g ${USER_GID} \
    -m \
    ${USER}

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && \
    chown -R ${USER}:${GROUP} /opt/python

WORKDIR /opt/tide2

RUN chown ${USER}:${GROUP} /opt/tide2 && \
    chmod 755 /opt/tide2

ENV TIDE_CACHE_DIR=/home/${USER}/.cache/tide
RUN mkdir -p ${TIDE_CACHE_DIR} && \
    chown -R ${USER}:${GROUP} /home/${USER}


# ============================================================================
# CPU-only production target (no GPU packages)
# ============================================================================
FROM cpu-base AS production-cpu

ARG USER
ARG GROUP
ARG USER_UID
ARG USER_GID

USER ${USER}

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/${USER}/.cache/uv \
    UV_INDEX_PRIVATE_REGISTRY_USERNAME=oauth2accesstoken \
    PATH="/home/${USER}/.local/bin:/opt/tide2/.venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/tide2/.venv

COPY --chown=${USER}:${GROUP} pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/home/${USER}/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --locked --no-install-project --no-dev

COPY --chown=${USER}:${GROUP} README.md LICENSE-MIT ./
COPY --chown=${USER}:${GROUP} src ./src

RUN --mount=type=cache,target=/home/${USER}/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --locked --no-dev

ARG DOCKER_REGISTRY=
ARG DOCKER_IMAGE_CPU=tide2-cpu
ARG DOCKER_IMAGE_GPU=tide2-gpu
ARG DOCKER_IMAGE_TAG=latest
ENV PATH="/opt/tide2/.venv/bin:$PATH" \
    DOCKER_REGISTRY=${DOCKER_REGISTRY} \
    DOCKER_IMAGE_CPU=${DOCKER_IMAGE_CPU} \
    DOCKER_IMAGE_GPU=${DOCKER_IMAGE_GPU} \
    DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG}

ENV UV_LOCKED=1 \
    UV_NO_SYNC=1 \
    UV_NO_CACHE=1 \
    UV_NO_DEV=1

COPY --chown=${USER}:${GROUP} prefect.yaml ./
COPY --chown=${USER}:${GROUP} prefect_job_template.json ./



# ============================================================================
# Production target (GPU)
# ============================================================================
FROM base AS production-gpu

ARG USER
ARG GROUP
ARG USER_UID
ARG USER_GID

USER ${USER}

# Configure uv environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/${USER}/.cache/uv \
    PATH="/home/${USER}/.local/bin:/opt/tide2/.venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/tide2/.venv

# Copy lockfiles so the dependency layer can be cached
COPY --chown=${USER}:${GROUP} pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/home/${USER}/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --locked --no-install-project --no-dev

# Copy project files
COPY --chown=${USER}:${GROUP} README.md LICENSE-MIT ./
COPY --chown=${USER}:${GROUP} src ./src

# Install the project
RUN --mount=type=cache,target=/home/${USER}/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --locked --no-dev

# NVIDIA runtime environment variables (mounted by GKE)
ARG DOCKER_REGISTRY=
ARG DOCKER_IMAGE_CPU=tide2-cpu
ARG DOCKER_IMAGE_GPU=tide2-gpu
ARG DOCKER_IMAGE_TAG=latest
ENV PATH="/opt/tide2/.venv/bin:/usr/local/nvidia/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64" \
    DOCKER_REGISTRY=${DOCKER_REGISTRY} \
    DOCKER_IMAGE_CPU=${DOCKER_IMAGE_CPU} \
    DOCKER_IMAGE_GPU=${DOCKER_IMAGE_GPU} \
    DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG}

ENV UV_LOCKED=1 \
    UV_NO_SYNC=1 \
    UV_NO_CACHE=1 \
    UV_NO_DEV=1

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY --chown=${USER}:${GROUP} prefect.yaml ./

# ============================================================================
# Development target
# ============================================================================
FROM base AS development

USER root

# Install development tools and Google Cloud SDK
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    git curl vim sudo gpg apt-transport-https ca-certificates \
    build-essential cmake swig \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin \
    && rm -rf /var/lib/apt/lists/*

ARG USER
ARG GROUP
ARG USER_UID
ARG USER_GID

RUN usermod -s /bin/bash ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER}

# Configure uv environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/.cache/uv \
    UV_LINK_MODE=copy \
    PATH="/home/${USER}/.local/bin:/opt/tide2/.venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/tide2/.venv

# Copy lockfiles so the dependency layer can be cached
COPY --chown=${USER}:${GROUP} pyproject.toml uv.lock ./

# Create uv cache directory and install dependencies
RUN mkdir -p ${UV_CACHE_DIR}
RUN --mount=type=cache,target=${UV_CACHE_DIR},uid=${USER_UID},gid=${USER_GID} \
    uv sync --locked --no-install-project --no-dev

ENV UV_CACHE_DIR=/home/${USER}/.cache/uv
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# =============================================================================
# Runs unit tests with coverage
# =============================================================================
FROM development AS test

ENV UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/tmp/.cache/uv \
    UV_LINK_MODE=copy

COPY --chown=${USER}:${GROUP} pyproject.toml uv.lock ./

RUN uv sync --no-install-project --frozen

COPY --chown=${USER}:${GROUP} README.md LICENSE-MIT ./
COPY --chown=${USER}:${GROUP} src ./src
COPY --chown=${USER}:${GROUP} tests ./tests
COPY --chown=${USER}:${GROUP} conftest.py ./

RUN uv sync --frozen

RUN uv run pytest --cov
