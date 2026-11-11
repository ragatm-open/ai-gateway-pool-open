# Qualification toolbox (NOT a service): the repository's tests + locked dev deps,
# run as a throw-away container on the pool's INTERNAL backend network to qualify
# the deployed staging Redis. Never started by Compose, never published.
# Qualification : même base épinglée par digest que l'image de service (voir ./Dockerfile).
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
FROM ${PYTHON_IMAGE}
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /repo
COPY requirements-dev.lock ./
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --require-hashes -r requirements-dev.lock
COPY . .
RUN /opt/venv/bin/pip install --no-deps --no-build-isolation -e .
RUN useradd --system --uid 10002 qual && chown -R qual /repo
USER 10002
ENV PATH=/opt/venv/bin:$PATH
ENTRYPOINT ["pytest", "-p", "no:cacheprovider"]
