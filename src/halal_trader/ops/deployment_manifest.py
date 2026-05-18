"""Cloud deployment manifest validator.

The roadmap pins Wave 3.I: "Dockerfile + docker-compose + Terraform
module + Helm chart for self-hosted multi-user. Documented
one-command deploy to AWS / GCP / Fly.io." This module is the
**pure-Python manifest spec + validator** that the actual
Dockerfile / docker-compose.yml / Helm chart / Terraform module
land separately and consume as their source of truth.

Picked a focused validator over hand-writing four separate
deployment artifacts because (a) the four targets share most of
their service topology (postgres + bot + dashboard) and only differ
in packaging — encoding the topology once means a service rename
or port change ripples to all four targets without manual sync,
(b) the secret-reference contract is the load-bearing safety
attribute (a docker-compose.yml with `DATABASE_URL: postgres://...:secret@host`
inline is a leaked-secret class of bug; encoding "secrets must be
references, never inline" as a validation rule makes the regression
structurally impossible), (c) resource-limit defaults (CPU / memory
per service) need consistent expression across targets — Terraform
takes `cpu_units`, K8s takes `cores`, docker-compose takes a
fraction; the manifest holds the canonical value and the per-target
serializer translates.

Pinned semantics:
- **Closed-set DeploymentTarget enum.** Adding a target is a code
  review change so the manifest serializer can't drift.
- **Required services for the bot to run.** `postgres` + `bot`
  + `dashboard` are the minimum; missing any is rejected.
- **Secrets are references, never inline.** `EnvVar.value` is
  None when `is_secret=True`; the manifest carries `secret_ref`
  (e.g. `${SECRETS_VAULT}/llm_api_key`) instead. Validation
  rejects any inline `value` on a secret EnvVar.
- **Resource limits enforced.** Memory in [128MB, 32GB], CPU
  in [0.1, 16.0] cores. Out-of-range fails fast.
- **Render output never includes inline secret values.** The
  serializer writes `secret_ref` placeholders only; the actual
  secret resolution is operator-side at deployment time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeploymentTarget(str, Enum):
    """Canonical deployment targets.

    Pinned string values for JSON / DB stability. Adding a target
    is a code review change.
    """

    DOCKER_COMPOSE = "docker_compose"
    K8S_HELM = "k8s_helm"
    TERRAFORM_AWS = "terraform_aws"
    TERRAFORM_GCP = "terraform_gcp"
    FLY_IO = "fly_io"


class ServiceKind(str, Enum):
    """Closed-set service categories.

    Pinned values. The bot deployment has three required service
    kinds; AUX is for optional add-ons (e.g. Grafana, Prometheus).
    """

    POSTGRES = "postgres"
    BOT = "bot"
    DASHBOARD = "dashboard"
    AUX = "aux"


_REQUIRED_KINDS: frozenset[ServiceKind] = frozenset(
    {ServiceKind.POSTGRES, ServiceKind.BOT, ServiceKind.DASHBOARD}
)


_MIN_MEMORY_MB = 128
_MAX_MEMORY_MB = 32_768
_MIN_CPU_CORES = 0.1
_MAX_CPU_CORES = 16.0


@dataclass(frozen=True)
class EnvVar:
    """One environment variable.

    Either `value` is set (for non-secret config) OR `secret_ref` is
    set (for secrets to be resolved at deployment time). Both inline
    and ref are mutually exclusive — pinned via validation.
    """

    name: str
    value: str | None = None
    secret_ref: str | None = None
    is_secret: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.is_secret:
            if self.value is not None:
                raise ValueError(
                    f"is_secret=True env {self.name!r} must not carry inline value (use secret_ref)"
                )
            if self.secret_ref is None or not self.secret_ref.strip():
                raise ValueError(f"is_secret=True env {self.name!r} requires secret_ref")
        else:
            if self.secret_ref is not None:
                raise ValueError(f"non-secret env {self.name!r} must not have secret_ref")
            if self.value is None:
                raise ValueError(f"non-secret env {self.name!r} requires inline value")


@dataclass(frozen=True)
class ResourceLimits:
    """Per-service resource limits."""

    memory_mb: int
    cpu_cores: float

    def __post_init__(self) -> None:
        if not _MIN_MEMORY_MB <= self.memory_mb <= _MAX_MEMORY_MB:
            raise ValueError(
                f"memory_mb {self.memory_mb} out of [{_MIN_MEMORY_MB}, {_MAX_MEMORY_MB}]"
            )
        if not _MIN_CPU_CORES <= self.cpu_cores <= _MAX_CPU_CORES:
            raise ValueError(
                f"cpu_cores {self.cpu_cores} out of [{_MIN_CPU_CORES}, {_MAX_CPU_CORES}]"
            )


@dataclass(frozen=True)
class ServiceSpec:
    """One service in the deployment manifest."""

    name: str
    kind: ServiceKind
    image: str
    env: tuple[EnvVar, ...]
    limits: ResourceLimits
    exposed_port: int | None = None  # None for internal-only services

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.image or not self.image.strip():
            raise ValueError("image must be non-empty")
        if self.exposed_port is not None:
            if not 1 <= self.exposed_port <= 65535:
                raise ValueError(f"exposed_port {self.exposed_port} out of [1, 65535]")
        # No duplicate env names within the service
        names = [e.name for e in self.env]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate env name in service {self.name!r}")


@dataclass(frozen=True)
class DeploymentManifest:
    """Cross-target deployment manifest.

    `target` is the deployment-target serializer this manifest will
    be rendered for. Validation enforces all required service kinds
    are present.
    """

    name: str
    version: str
    target: DeploymentTarget
    services: tuple[ServiceSpec, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.version or not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.services:
            raise ValueError("services must be non-empty")
        # No duplicate service names
        service_names = [s.name for s in self.services]
        if len(set(service_names)) != len(service_names):
            raise ValueError("duplicate service name")


class ManifestViolationError(Exception):
    """Raised when a manifest fails validation."""

    def __init__(self, manifest_name: str, reason: str) -> None:
        super().__init__(f"manifest {manifest_name!r}: {reason}")
        self.manifest_name = manifest_name
        self.reason = reason


def validate_manifest(manifest: DeploymentManifest) -> None:
    """Run all validation rules; raise ManifestViolationError on first failure.

    Operators call this before serializing the manifest to a target
    (Dockerfile / docker-compose / Helm / Terraform); a pre-serialize
    failure prevents shipping a broken deployment artifact.
    """

    # All required service kinds present
    present_kinds = {s.kind for s in manifest.services}
    missing = _REQUIRED_KINDS - present_kinds
    if missing:
        missing_str = ", ".join(sorted(k.value for k in missing))
        raise ManifestViolationError(
            manifest.name, f"missing required service kinds: {missing_str}"
        )

    # POSTGRES service must expose 5432 (the canonical Postgres port —
    # operators sometimes accidentally expose 5433 which is the test DB
    # port; the manifest is for production deploys)
    postgres_services = [s for s in manifest.services if s.kind is ServiceKind.POSTGRES]
    for s in postgres_services:
        if s.exposed_port is not None and s.exposed_port != 5432:
            raise ManifestViolationError(
                manifest.name,
                f"postgres service {s.name!r} exposes port "
                f"{s.exposed_port}; must be 5432 or None (internal-only)",
            )

    # BOT service must have a DATABASE_URL secret reference (not inline)
    bot_services = [s for s in manifest.services if s.kind is ServiceKind.BOT]
    for s in bot_services:
        env_by_name = {e.name: e for e in s.env}
        db_url = env_by_name.get("DATABASE_URL")
        if db_url is None:
            raise ManifestViolationError(
                manifest.name,
                f"bot service {s.name!r} missing DATABASE_URL env",
            )
        if not db_url.is_secret:
            raise ManifestViolationError(
                manifest.name,
                f"bot service {s.name!r} DATABASE_URL must be a secret "
                f"(use secret_ref, not inline value)",
            )

    # No duplicate service kinds where it matters (only one BOT per manifest)
    bot_count = sum(1 for s in manifest.services if s.kind is ServiceKind.BOT)
    if bot_count > 1:
        raise ManifestViolationError(
            manifest.name, f"only one BOT service allowed; found {bot_count}"
        )

    # The manifest's target must be a known target (covered structurally
    # by the enum, but we re-check in case of bypass).
    if not isinstance(manifest.target, DeploymentTarget):
        raise ManifestViolationError(
            manifest.name, f"unknown deployment target {manifest.target!r}"
        )


def collect_secret_refs(manifest: DeploymentManifest) -> tuple[str, ...]:
    """Return all secret_ref values in the manifest (deterministic order).

    Operators use this to verify the deployment-time secret-resolution
    plan covers every reference; a missing secret at deployment time
    means the bot fails to start.
    """

    refs: list[str] = []
    for service in manifest.services:
        for env_var in service.env:
            if env_var.is_secret and env_var.secret_ref is not None:
                refs.append(env_var.secret_ref)
    return tuple(sorted(set(refs)))


def total_resource_request(
    manifest: DeploymentManifest,
) -> tuple[int, float]:
    """Return (total_memory_mb, total_cpu_cores) summed across services."""

    total_mem = sum(s.limits.memory_mb for s in manifest.services)
    total_cpu = sum(s.limits.cpu_cores for s in manifest.services)
    return total_mem, total_cpu


_KIND_EMOJI: dict[ServiceKind, str] = {
    ServiceKind.POSTGRES: "🗄️",
    ServiceKind.BOT: "🤖",
    ServiceKind.DASHBOARD: "📊",
    ServiceKind.AUX: "🧰",
}


_TARGET_EMOJI: dict[DeploymentTarget, str] = {
    DeploymentTarget.DOCKER_COMPOSE: "🐳",
    DeploymentTarget.K8S_HELM: "☸️",
    DeploymentTarget.TERRAFORM_AWS: "🟧",
    DeploymentTarget.TERRAFORM_GCP: "🔷",
    DeploymentTarget.FLY_IO: "🪂",
}


def render_service(service: ServiceSpec) -> str:
    """Format one service for ops display.

    No-secret-leak: secret env vars render their `secret_ref`
    placeholder only; never the resolved value.
    """

    emoji = _KIND_EMOJI[service.kind]
    port_str = f":{service.exposed_port}" if service.exposed_port else " (internal)"
    lines = [
        f"{emoji} {service.name} ({service.kind.value}){port_str}",
        f"  image: {service.image}",
        f"  resources: {service.limits.memory_mb}MB / {service.limits.cpu_cores} CPU",
    ]
    if service.env:
        lines.append("  env:")
        for env_var in sorted(service.env, key=lambda e: e.name):
            if env_var.is_secret:
                lines.append(f"    {env_var.name}: <secret:{env_var.secret_ref}>")
            else:
                lines.append(f"    {env_var.name}: {env_var.value}")
    return "\n".join(lines)


def render_manifest(manifest: DeploymentManifest) -> str:
    """Format the manifest for ops display."""

    target_emoji = _TARGET_EMOJI[manifest.target]
    total_mem, total_cpu = total_resource_request(manifest)
    lines = [
        f"{target_emoji} {manifest.name} v{manifest.version} → {manifest.target.value}",
        f"  services: {len(manifest.services)}",
        f"  total: {total_mem}MB / {total_cpu:.1f} CPU",
    ]
    for service in manifest.services:
        lines.append("")
        lines.append(render_service(service))
    return "\n".join(lines)


__all__ = [
    "DeploymentManifest",
    "DeploymentTarget",
    "EnvVar",
    "ManifestViolationError",
    "ResourceLimits",
    "ServiceKind",
    "ServiceSpec",
    "collect_secret_refs",
    "render_manifest",
    "render_service",
    "total_resource_request",
    "validate_manifest",
]
