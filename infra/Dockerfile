# Halal Trader runtime image — Python 3.14 + uv-managed dependencies.
#
# Three-stage build:
#   1. spa-builder  — Node 22 builds the React dashboard (Vite)
#   2. py-builder   — Python 3.14 + uv resolves the venv
#   3. runtime      — slim image with venv + source + dashboard/dist
#
# The dashboard is built inside the image so the runtime is fully
# self-contained — no host bind-mount required. uv is kept in the
# runtime stage because `uvx` launches the Alpaca MCP subprocess.

# ── Stage 1: dashboard SPA ──
FROM node:22-alpine AS spa-builder

WORKDIR /spa

# Lock-file first for cache reuse.
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY dashboard/ ./
RUN npm run build

# ── Stage 2: Python venv ──
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# uv installs into ~/.local/bin; expose it on PATH for the build steps.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first so a code change doesn't bust the
# layer cache for sync — uv reads pyproject.toml + uv.lock to decide
# what to install. README.md is referenced by pyproject's [project]
# table so it must be present at sync time.
COPY pyproject.toml uv.lock README.md ./

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
    PATH="/app/.venv/bin:/usr/local/bin:${PATH}"

WORKDIR /app

# Bring the venv + project source in. We also need `uvx` at runtime
# because the Alpaca MCP adapter launches its server via `uvx`; install
# uv (which provides uvx) into the final image.
RUN pip install --no-cache-dir uv

COPY --from=builder /app /app

# Drop the built SPA into the runtime image where web/app.py looks for it
# (../../../dashboard/dist relative to web/app.py → /app/dashboard/dist).
COPY --from=spa-builder /spa/dist /app/dashboard/dist

# Non-root user so a container escape doesn't land you on root.
# We need a real home dir for `uvx` to write its cache when launching
# the Alpaca MCP subprocess — bot otherwise crashes on stocks startup.
RUN useradd --create-home --uid 1000 trader \
    && chown -R trader:trader /app /home/trader
USER trader
ENV HOME=/home/trader \
    UV_CACHE_DIR=/home/trader/.cache/uv

# Default to the crypto bot; override at run time for stocks or web.
CMD ["halal-trader", "crypto", "start"]
