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

# Sync core deps; the dashboard / ml / sentiment extras are optional and the
# operator can rebuild with --build-arg EXTRAS="ml,sentiment". EXTRAS is
# comma-separated, but uv's --extra takes ONE name per flag — so expand it into
# repeated --extra flags. Passing "ml,sentiment" as a single --extra value
# silently matches no extra and installs neither (base deps only). When EXTRAS
# is empty the loop adds nothing and this is a plain `uv sync --frozen`.
ARG EXTRAS=""
RUN set --; \
    for e in $(echo "$EXTRAS" | tr ',' ' '); do set -- "$@" --extra "$e"; done; \
    uv sync --frozen "$@"

COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Re-sync after copying the source so the project package itself is installed.
# The extras MUST be repeated here — a bare `uv sync` would prune them back out.
RUN set --; \
    for e in $(echo "$EXTRAS" | tr ',' ' '); do set -- "$@" --extra "$e"; done; \
    uv sync --frozen "$@"

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
#
# /app/logs and /app/data are declared + owned by `trader` here so that when
# the compose `trader-logs`/`trader-data` named volumes mount over them, Docker
# seeds the (empty) volumes with `trader` ownership. Without these dirs existing
# in the image, a fresh named volume is created root-owned and the non-root
# process hits `PermissionError` on the first log write (crash-loops on start).
RUN useradd --create-home --uid 1000 trader \
    && mkdir -p /app/logs /app/data \
    && chown -R trader:trader /app /home/trader
USER trader
ENV HOME=/home/trader \
    UV_CACHE_DIR=/home/trader/.cache/uv

# Default to the crypto bot; override at run time for stocks or web.
CMD ["halal-trader", "crypto", "start"]
