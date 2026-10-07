# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# agent-service image. Multi-stage build:
#   deps     -> Python packages in a virtualenv
#   test     -> deps + dev tools; `docker build --target test .` runs pytest
#   runtime  -> slim final image, non-root, no compilers, no tests
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
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
LABEL org.opencontainers.image.title="documind-agent-service" \
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
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", \
     "--timeout-graceful-shutdown", "25", "--no-access-log"]
