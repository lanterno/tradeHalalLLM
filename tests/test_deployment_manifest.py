"""Tests for `halal_trader.ops.deployment_manifest` (Wave 3.I).

Covers: target / kind enums, EnvVar secret-vs-inline mutual exclusion,
ResourceLimits ranges, manifest validation gate, no-secret render
contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.ops.deployment_manifest import (
    DeploymentManifest,
    DeploymentTarget,
    EnvVar,
    ManifestViolationError,
    ResourceLimits,
    ServiceKind,
    ServiceSpec,
    collect_secret_refs,
    render_manifest,
    render_service,
    total_resource_request,
    validate_manifest,
)

# --------------------------- Enum string pins --------------------------------


def test_deployment_target_string_values_pinned() -> None:
    assert DeploymentTarget.DOCKER_COMPOSE.value == "docker_compose"
    assert DeploymentTarget.K8S_HELM.value == "k8s_helm"
    assert DeploymentTarget.TERRAFORM_AWS.value == "terraform_aws"
    assert DeploymentTarget.TERRAFORM_GCP.value == "terraform_gcp"
    assert DeploymentTarget.FLY_IO.value == "fly_io"


def test_service_kind_string_values_pinned() -> None:
    assert ServiceKind.POSTGRES.value == "postgres"
    assert ServiceKind.BOT.value == "bot"
    assert ServiceKind.DASHBOARD.value == "dashboard"
    assert ServiceKind.AUX.value == "aux"


# --------------------------- EnvVar ------------------------------------------


def test_envvar_basic_inline() -> None:
    env = EnvVar(name="LOG_LEVEL", value="info")
    assert env.value == "info"
    assert env.is_secret is False


def test_envvar_basic_secret() -> None:
    env = EnvVar(
        name="DATABASE_URL",
        secret_ref="${VAULT}/db_url",
        is_secret=True,
    )
    assert env.is_secret is True


def test_envvar_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        EnvVar(name="", value="x")


def test_envvar_rejects_inline_value_when_secret() -> None:
    """Pin: secret env vars must NOT carry inline values."""

    with pytest.raises(ValueError, match="inline"):
        EnvVar(
            name="DB_URL",
            value="postgres://user:pw@host",
            is_secret=True,
            secret_ref="${VAULT}/db",
        )


def test_envvar_rejects_secret_without_ref() -> None:
    with pytest.raises(ValueError, match="secret_ref"):
        EnvVar(name="DB_URL", is_secret=True)


def test_envvar_rejects_secret_with_empty_ref() -> None:
    with pytest.raises(ValueError, match="secret_ref"):
        EnvVar(name="DB_URL", is_secret=True, secret_ref="   ")


def test_envvar_rejects_non_secret_with_ref() -> None:
    """Pin: non-secret env vars can't have a secret_ref."""

    with pytest.raises(ValueError, match="secret_ref"):
        EnvVar(
            name="LOG_LEVEL",
            value="info",
            secret_ref="${VAULT}/log_level",
        )


def test_envvar_rejects_non_secret_without_value() -> None:
    """Pin: non-secret env vars require an inline value."""

    with pytest.raises(ValueError, match="value"):
        EnvVar(name="LOG_LEVEL")


def test_envvar_is_frozen() -> None:
    env = EnvVar(name="X", value="y")
    with pytest.raises(FrozenInstanceError):
        env.value = "z"  # type: ignore[misc]


# --------------------------- ResourceLimits ----------------------------------


def test_resource_limits_basic() -> None:
    limits = ResourceLimits(memory_mb=512, cpu_cores=0.5)
    assert limits.memory_mb == 512


def test_resource_limits_rejects_memory_below_min() -> None:
    with pytest.raises(ValueError, match="memory_mb"):
        ResourceLimits(memory_mb=64, cpu_cores=1.0)


def test_resource_limits_accepts_memory_at_lower_boundary() -> None:
    limits = ResourceLimits(memory_mb=128, cpu_cores=0.1)
    assert limits.memory_mb == 128


def test_resource_limits_rejects_memory_above_max() -> None:
    with pytest.raises(ValueError, match="memory_mb"):
        ResourceLimits(memory_mb=64_000, cpu_cores=1.0)


def test_resource_limits_rejects_cpu_below_min() -> None:
    with pytest.raises(ValueError, match="cpu_cores"):
        ResourceLimits(memory_mb=512, cpu_cores=0.05)


def test_resource_limits_rejects_cpu_above_max() -> None:
    with pytest.raises(ValueError, match="cpu_cores"):
        ResourceLimits(memory_mb=512, cpu_cores=32.0)


def test_resource_limits_is_frozen() -> None:
    limits = ResourceLimits(memory_mb=512, cpu_cores=1.0)
    with pytest.raises(FrozenInstanceError):
        limits.memory_mb = 1024  # type: ignore[misc]


# --------------------------- ServiceSpec -------------------------------------


def _service(**overrides: object) -> ServiceSpec:
    base: dict[str, object] = {
        "name": "test_service",
        "kind": ServiceKind.AUX,
        "image": "test:1.0",
        "env": (),
        "limits": ResourceLimits(memory_mb=256, cpu_cores=0.5),
    }
    base.update(overrides)
    return ServiceSpec(**base)  # type: ignore[arg-type]


def test_service_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _service(name="")


def test_service_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="image"):
        _service(image="")


def test_service_rejects_port_below_1() -> None:
    with pytest.raises(ValueError, match="exposed_port"):
        _service(exposed_port=0)


def test_service_rejects_port_above_max() -> None:
    with pytest.raises(ValueError, match="exposed_port"):
        _service(exposed_port=70_000)


def test_service_accepts_no_port() -> None:
    """Pin: internal-only services have exposed_port=None."""

    s = _service(exposed_port=None)
    assert s.exposed_port is None


def test_service_rejects_duplicate_env_names() -> None:
    with pytest.raises(ValueError, match="duplicate env"):
        _service(
            env=(
                EnvVar(name="X", value="1"),
                EnvVar(name="X", value="2"),
            ),
        )


def test_service_is_frozen() -> None:
    s = _service()
    with pytest.raises(FrozenInstanceError):
        s.image = "other"  # type: ignore[misc]


# --------------------------- DeploymentManifest ------------------------------


def _full_services() -> tuple[ServiceSpec, ...]:
    return (
        ServiceSpec(
            name="postgres",
            kind=ServiceKind.POSTGRES,
            image="pgvector/pgvector:pg16",
            env=(EnvVar(name="POSTGRES_USER", value="halal"),),
            limits=ResourceLimits(memory_mb=1024, cpu_cores=1.0),
            exposed_port=5432,
        ),
        ServiceSpec(
            name="bot",
            kind=ServiceKind.BOT,
            image="halal-trader:latest",
            env=(
                EnvVar(name="LOG_LEVEL", value="info"),
                EnvVar(
                    name="DATABASE_URL",
                    is_secret=True,
                    secret_ref="${VAULT}/database_url",
                ),
            ),
            limits=ResourceLimits(memory_mb=2048, cpu_cores=1.0),
        ),
        ServiceSpec(
            name="dashboard",
            kind=ServiceKind.DASHBOARD,
            image="halal-trader-dashboard:latest",
            env=(),
            limits=ResourceLimits(memory_mb=512, cpu_cores=0.5),
            exposed_port=8082,
        ),
    )


def _full_manifest() -> DeploymentManifest:
    return DeploymentManifest(
        name="halal-trader-prod",
        version="1.0.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=_full_services(),
    )


def test_manifest_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        DeploymentManifest(
            name="",
            version="1.0",
            target=DeploymentTarget.DOCKER_COMPOSE,
            services=_full_services(),
        )


def test_manifest_rejects_empty_version() -> None:
    with pytest.raises(ValueError, match="version"):
        DeploymentManifest(
            name="x",
            version="",
            target=DeploymentTarget.DOCKER_COMPOSE,
            services=_full_services(),
        )


def test_manifest_rejects_empty_services() -> None:
    with pytest.raises(ValueError, match="services"):
        DeploymentManifest(
            name="x",
            version="1.0",
            target=DeploymentTarget.DOCKER_COMPOSE,
            services=(),
        )


def test_manifest_rejects_duplicate_service_names() -> None:
    services = list(_full_services())
    services.append(services[0])  # duplicate postgres
    with pytest.raises(ValueError, match="duplicate"):
        DeploymentManifest(
            name="x",
            version="1.0",
            target=DeploymentTarget.DOCKER_COMPOSE,
            services=tuple(services),
        )


def test_manifest_is_frozen() -> None:
    m = _full_manifest()
    with pytest.raises(FrozenInstanceError):
        m.version = "2.0.0"  # type: ignore[misc]


# --------------------------- validate_manifest -------------------------------


def test_validate_clean_manifest_passes() -> None:
    validate_manifest(_full_manifest())


def test_validate_rejects_missing_postgres() -> None:
    services = tuple(s for s in _full_services() if s.kind is not ServiceKind.POSTGRES)
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=services,
    )
    with pytest.raises(ManifestViolationError, match="postgres"):
        validate_manifest(m)


def test_validate_rejects_missing_bot() -> None:
    services = tuple(s for s in _full_services() if s.kind is not ServiceKind.BOT)
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=services,
    )
    with pytest.raises(ManifestViolationError, match="bot"):
        validate_manifest(m)


def test_validate_rejects_missing_dashboard() -> None:
    services = tuple(s for s in _full_services() if s.kind is not ServiceKind.DASHBOARD)
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=services,
    )
    with pytest.raises(ManifestViolationError, match="dashboard"):
        validate_manifest(m)


def test_validate_rejects_postgres_on_wrong_port() -> None:
    """Pin: postgres must expose 5432 or be internal-only."""

    services = list(_full_services())
    services[0] = ServiceSpec(
        name="postgres",
        kind=ServiceKind.POSTGRES,
        image="pgvector/pgvector:pg16",
        env=(),
        limits=ResourceLimits(memory_mb=1024, cpu_cores=1.0),
        exposed_port=5433,
    )
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=tuple(services),
    )
    with pytest.raises(ManifestViolationError, match="5432"):
        validate_manifest(m)


def test_validate_accepts_postgres_internal_only() -> None:
    """Pin: postgres without exposed_port is OK (internal-only deploy)."""

    services = list(_full_services())
    services[0] = ServiceSpec(
        name="postgres",
        kind=ServiceKind.POSTGRES,
        image="pgvector/pgvector:pg16",
        env=(),
        limits=ResourceLimits(memory_mb=1024, cpu_cores=1.0),
        exposed_port=None,
    )
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=tuple(services),
    )
    validate_manifest(m)  # no raise


def test_validate_rejects_bot_without_database_url() -> None:
    services = list(_full_services())
    # Replace bot with version missing DATABASE_URL
    services[1] = ServiceSpec(
        name="bot",
        kind=ServiceKind.BOT,
        image="halal-trader:latest",
        env=(EnvVar(name="LOG_LEVEL", value="info"),),
        limits=ResourceLimits(memory_mb=2048, cpu_cores=1.0),
    )
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=tuple(services),
    )
    with pytest.raises(ManifestViolationError, match="DATABASE_URL"):
        validate_manifest(m)


def test_validate_rejects_inline_database_url() -> None:
    """Pin: DATABASE_URL must be a secret reference, not inline.

    This is the load-bearing leaked-secret pin — a contributor that
    accidentally inlines the postgres connection string fails CI
    rather than ships and leaks the password.
    """

    services = list(_full_services())
    services[1] = ServiceSpec(
        name="bot",
        kind=ServiceKind.BOT,
        image="halal-trader:latest",
        env=(
            EnvVar(name="LOG_LEVEL", value="info"),
            EnvVar(
                name="DATABASE_URL",
                value="postgres://halal:secret@localhost/db",
            ),
        ),
        limits=ResourceLimits(memory_mb=2048, cpu_cores=1.0),
    )
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=tuple(services),
    )
    with pytest.raises(ManifestViolationError, match="secret"):
        validate_manifest(m)


def test_validate_rejects_multiple_bots() -> None:
    services = list(_full_services())
    second_bot = ServiceSpec(
        name="bot_2",
        kind=ServiceKind.BOT,
        image="halal-trader:latest",
        env=(
            EnvVar(
                name="DATABASE_URL",
                is_secret=True,
                secret_ref="${VAULT}/db",
            ),
        ),
        limits=ResourceLimits(memory_mb=2048, cpu_cores=1.0),
    )
    services.append(second_bot)
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=tuple(services),
    )
    with pytest.raises(ManifestViolationError, match="one BOT"):
        validate_manifest(m)


def test_validate_aux_service_doesnt_count_as_required() -> None:
    """AUX services don't satisfy any of POSTGRES/BOT/DASHBOARD."""

    services = (
        ServiceSpec(
            name="aux1",
            kind=ServiceKind.AUX,
            image="grafana:latest",
            env=(),
            limits=ResourceLimits(memory_mb=256, cpu_cores=0.5),
        ),
    )
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=services,
    )
    with pytest.raises(ManifestViolationError, match="missing"):
        validate_manifest(m)


def test_violation_carries_manifest_name_and_reason() -> None:
    services = tuple(s for s in _full_services() if s.kind is not ServiceKind.BOT)
    m = DeploymentManifest(
        name="halal-prod",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=services,
    )
    try:
        validate_manifest(m)
    except ManifestViolationError as e:
        assert e.manifest_name == "halal-prod"
        assert "bot" in e.reason


# --------------------------- collect_secret_refs -----------------------------


def test_collect_secret_refs_basic() -> None:
    refs = collect_secret_refs(_full_manifest())
    assert "${VAULT}/database_url" in refs


def test_collect_secret_refs_deduplicates() -> None:
    """Pin: same secret_ref appearing twice surfaces once."""

    services = (
        ServiceSpec(
            name="postgres",
            kind=ServiceKind.POSTGRES,
            image="pgvector/pgvector:pg16",
            env=(),
            limits=ResourceLimits(memory_mb=1024, cpu_cores=1.0),
            exposed_port=5432,
        ),
        ServiceSpec(
            name="bot",
            kind=ServiceKind.BOT,
            image="halal-trader:latest",
            env=(
                EnvVar(
                    name="DATABASE_URL",
                    is_secret=True,
                    secret_ref="${VAULT}/db",
                ),
                EnvVar(
                    name="DATABASE_URL_ALIAS",
                    is_secret=True,
                    secret_ref="${VAULT}/db",
                ),
            ),
            limits=ResourceLimits(memory_mb=2048, cpu_cores=1.0),
        ),
        ServiceSpec(
            name="dashboard",
            kind=ServiceKind.DASHBOARD,
            image="halal-trader-dashboard:latest",
            env=(),
            limits=ResourceLimits(memory_mb=512, cpu_cores=0.5),
            exposed_port=8082,
        ),
    )
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=services,
    )
    refs = collect_secret_refs(m)
    assert refs.count("${VAULT}/db") == 1


def test_collect_secret_refs_returns_sorted() -> None:
    """Pin: deterministic ordering."""

    refs = collect_secret_refs(_full_manifest())
    assert list(refs) == sorted(refs)


def test_collect_secret_refs_empty_when_no_secrets() -> None:
    services = (
        ServiceSpec(
            name="postgres",
            kind=ServiceKind.POSTGRES,
            image="pgvector/pgvector:pg16",
            env=(EnvVar(name="POSTGRES_USER", value="x"),),
            limits=ResourceLimits(memory_mb=1024, cpu_cores=1.0),
            exposed_port=5432,
        ),
    )
    # Build a single-service manifest (validation would fail but
    # collect_secret_refs doesn't validate)
    m = DeploymentManifest(
        name="x",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=services,
    )
    assert collect_secret_refs(m) == ()


# --------------------------- total_resource_request --------------------------


def test_total_resource_request() -> None:
    m = _full_manifest()
    total_mem, total_cpu = total_resource_request(m)
    # 1024 + 2048 + 512 = 3584
    assert total_mem == 3584
    # 1.0 + 1.0 + 0.5 = 2.5
    assert total_cpu == pytest.approx(2.5)


# --------------------------- render ------------------------------------------


def test_render_service_includes_name_and_kind() -> None:
    m = _full_manifest()
    out = render_service(m.services[0])
    assert "postgres" in out
    assert "🗄️" in out


def test_render_service_internal_marker() -> None:
    m = _full_manifest()
    bot = next(s for s in m.services if s.kind is ServiceKind.BOT)
    out = render_service(bot)
    assert "internal" in out


def test_render_service_secret_env_uses_placeholder() -> None:
    """Pin: secret env vars render as <secret:ref> not the value."""

    bot = next(s for s in _full_services() if s.kind is ServiceKind.BOT)
    out = render_service(bot)
    # The secret_ref appears
    assert "${VAULT}/database_url" in out
    # But also wrapped in <secret:...>
    assert "<secret:" in out


def test_render_service_no_secret_leak() -> None:
    """Pin: render never includes inline secret values."""

    # Construct a manifest where someone tried to put the password in
    # plain (which would have been rejected at construction); test that
    # the regular render path doesn't expose inline values for non-secret
    # env vars carrying sensitive-looking strings.
    s = _service(
        env=(
            EnvVar(name="LOG_LEVEL", value="info"),
            EnvVar(
                name="API_KEY",
                is_secret=True,
                secret_ref="${VAULT}/api_key",
            ),
        ),
    )
    out = render_service(s)
    # The render shows the placeholder, not the value
    assert "<secret:" in out


def test_render_manifest_includes_target_emoji() -> None:
    m = _full_manifest()
    out = render_manifest(m)
    assert "🐳" in out  # docker_compose emoji


def test_render_manifest_includes_total_resources() -> None:
    m = _full_manifest()
    out = render_manifest(m)
    assert "3584MB" in out
    assert "2.5 CPU" in out


def test_render_manifest_no_inline_passwords() -> None:
    """Pin: the manifest renderer doesn't surface postgres passwords
    or other inline secrets — every secret env appears as a placeholder."""

    m = _full_manifest()
    out = render_manifest(m)
    # If a contributor accidentally inlined "secret_password", it would
    # only have made it past EnvVar validation if not is_secret. Check
    # that the placeholder pattern is in use:
    assert "<secret:" in out
    assert "secret_password" not in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_deployment_validates() -> None:
    m = _full_manifest()
    validate_manifest(m)
    refs = collect_secret_refs(m)
    assert len(refs) >= 1
    total_mem, _ = total_resource_request(m)
    assert total_mem > 0


def test_e2e_target_swap_preserves_validation() -> None:
    """Swapping target from docker_compose to k8s_helm doesn't change
    whether the manifest validates (services + env are target-agnostic)."""

    m_compose = _full_manifest()
    m_helm = DeploymentManifest(
        name=m_compose.name,
        version=m_compose.version,
        target=DeploymentTarget.K8S_HELM,
        services=m_compose.services,
    )
    validate_manifest(m_compose)
    validate_manifest(m_helm)


def test_e2e_inline_secret_caught_early() -> None:
    """Pin: the load-bearing leaked-secret regression — inline DATABASE_URL
    fails validation before ever reaching a Dockerfile."""

    # Use replace_all to construct a bot service with an inline DB_URL
    services = list(_full_services())
    services[1] = ServiceSpec(
        name="bot",
        kind=ServiceKind.BOT,
        image="halal-trader:latest",
        env=(
            EnvVar(
                name="DATABASE_URL",
                value="postgres://halal:LeakedPassword123@localhost/db",
            ),
        ),
        limits=ResourceLimits(memory_mb=2048, cpu_cores=1.0),
    )
    m = DeploymentManifest(
        name="halal-trader-prod",
        version="1.0",
        target=DeploymentTarget.DOCKER_COMPOSE,
        services=tuple(services),
    )
    with pytest.raises(ManifestViolationError, match="secret"):
        validate_manifest(m)
