# AI Gateway Pool — non-GPU Python service.
# Build:  docker build --build-arg APP_GIT_SHA=$(git rev-parse HEAD) -t ai-gateway-pool:<sha> .
# The image contains NO configuration: the pool YAML and env are mounted per environment.

# Base épinglée par DIGEST : liste de manifestes multi-arch `python:3.12-slim` (Debian 13 « trixie »), relevée le
# 2026-11-09 — même base que les images ragatm-open-backend et ai-gateway-open. La base Debian 12 précédente portait
# deux avis CRITIQUES sans correctif (libsqlite3-0, zlib1g). Une mise à jour de la base est un changement explicite de
# ce fichier, jamais un glissement du tag.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY requirements.lock ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --require-hashes --no-deps -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
# The project wheel itself carries no dependency: runtime deps come ONLY from the hashed lock.
RUN /opt/venv/bin/pip wheel --no-deps -w /wheels . \
 && /opt/venv/bin/pip install --no-deps /wheels/*.whl

FROM ${PYTHON_IMAGE} AS runtime
ARG APP_GIT_SHA=unknown
LABEL org.opencontainers.image.title="ai-gateway-pool" \
      org.opencontainers.image.revision="${APP_GIT_SHA}" \
      org.opencontainers.image.description="Router in front of AI Gateway nodes (no GPU)"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_GIT_SHA=${APP_GIT_SHA} \
    POOL_BIND_HOST=0.0.0.0 \
    POOL_BIND_PORT=8080
RUN groupadd --system --gid 10001 pool \
 && useradd --system --uid 10001 --gid pool --no-create-home --shell /usr/sbin/nologin pool \
 && echo "${APP_GIT_SHA}" > /BUILD_REVISION
COPY --from=build /opt/venv /opt/venv
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2).status == 200 else 1)"]
# Exec form: the Python process is PID 1 and receives SIGTERM for a graceful uvicorn shutdown.
STOPSIGNAL SIGTERM
CMD ["ai-gateway-pool"]
