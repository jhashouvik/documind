# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# knowledge-service image. Multi-stage build:
#   deps     -> installs Python packages into a virtualenv
#   test     -> deps + dev tools + tests; build it with --target test to run
#               the unit tests inside Docker (no Python needed on the VM)
#   runtime  -> slim final image: venv + app code, runs as non-root
# Embeddings come from the OpenRouter API, so no model is baked in: the
# image is small (~250 MB) and pods start in about two seconds.
# BuildKit only builds the stages the chosen target needs, so a normal
# `docker build` never runs the test stage.
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
# requirements first: this layer is cached until requirements.txt changes,
# so editing app code does not reinstall every package
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM deps AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
WORKDIR /src
COPY . .
RUN python -m pytest

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG APP_VERSION=1.0.0
LABEL org.opencontainers.image.title="documind-knowledge-service" \
      org.opencontainers.image.version="${APP_VERSION}"
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=${APP_VERSION}
COPY --from=deps /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app ./app
USER 10001
EXPOSE 8001
# exec form: uvicorn is PID 1 and receives SIGTERM directly from Kubernetes
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", \
     "--timeout-graceful-shutdown", "20", "--no-access-log"]
