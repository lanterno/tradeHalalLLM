# Halal Trader runtime image — Python 3.14 + uv-managed dependencies.
#
# Multi-stage build keeps the final image lean: stage 1 fetches
# dependencies into a virtualenv with uv; stage 2 copies the venv +
# source into a slim runtime image. uv is a build-time dependency
# only, not runtime, so we don't ship its 30MB binary in production.
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# uv installs into ~/.local/bin; expose it on PATH for the build steps.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first so a code change doesn't bust the
# layer cache for sync — uv reads pyproject.toml + uv.lock to decide
# what to install.
COPY pyproject.toml uv.lock ./

# Sync core deps; the dashboard / ml / sentiment extras are optional
# and the operator can rebuild with --build-arg EXTRAS="ml,sentiment".
ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then \
      uv sync --frozen --extra $EXTRAS; \
    else \
      uv sync --frozen; \
    fi

COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

RUN uv sync --frozen

FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Bring just the venv + project source in — no pip / uv in the final image.
COPY --from=builder /app /app

# Non-root user so a container escape doesn't land you on root.
RUN useradd --no-create-home --uid 1000 trader \
    && chown -R trader:trader /app
USER trader

# Default to the crypto bot; override at run time for stocks or web.
CMD ["halal-trader", "crypto", "start"]
