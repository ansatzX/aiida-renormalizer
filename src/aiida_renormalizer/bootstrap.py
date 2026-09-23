"""Non-interactive AiiDA bootstrap for local calcfunction provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiida.manage import get_manager
from aiida.manage.configuration import create_profile, get_config

LOCAL_STORAGE_BACKEND = "core.sqlite_dos"


class BootstrapConfigurationError(ValueError):
    """Raised when an existing profile conflicts with the requested setup."""


@dataclass(frozen=True)
class ProfileBootstrapReport:
    """Observable result of creating or validating a local AiiDA profile."""

    profile_name: str
    profile_uuid: str
    created: bool
    config_dir: str
    storage_backend: str
    storage_path: str
    broker_backend: str | None
    default_user_email: str | None
    default_profile: bool
    loaded: bool


def _default_storage_path(config_dir: str, profile_name: str) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in profile_name
    )
    return (Path(config_dir) / "repository" / f"sqlite_dos_{safe_name}").resolve()


def _validate_existing_profile(profile, requested_storage_path: Path | None) -> Path:
    if profile.storage_backend != LOCAL_STORAGE_BACKEND:
        raise BootstrapConfigurationError(
            f"profile {profile.name!r} uses storage backend {profile.storage_backend!r}; "
            f"expected {LOCAL_STORAGE_BACKEND!r}"
        )

    try:
        actual_storage_path = Path(profile.storage_config["filepath"]).resolve()
    except KeyError as exception:
        raise BootstrapConfigurationError(
            f"profile {profile.name!r} has no sqlite storage filepath"
        ) from exception

    if requested_storage_path is not None and actual_storage_path != requested_storage_path:
        raise BootstrapConfigurationError(
            f"profile {profile.name!r} already uses storage path {actual_storage_path}; "
            f"requested {requested_storage_path}"
        )
    return actual_storage_path


def _load_profile_without_implicit_switch(profile) -> None:
    manager = get_manager()
    loaded_profile = manager.get_profile()
    if loaded_profile is not None and loaded_profile.uuid != profile.uuid:
        raise BootstrapConfigurationError(
            f"profile {loaded_profile.name!r} is already loaded in this Python process; "
            f"start a new process before loading {profile.name!r}"
        )
    manager.load_profile(profile)
    # Force storage construction now so setup failures are visible here, not in a later example.
    manager.get_profile_storage()


def _validate_profile_load_request(config, profile_name: str) -> None:
    loaded_profile = get_manager().get_profile()
    if loaded_profile is None:
        return
    if profile_name in config.profile_names:
        requested_profile = config.get_profile(profile_name)
        if requested_profile.uuid == loaded_profile.uuid:
            return
    raise BootstrapConfigurationError(
        f"profile {loaded_profile.name!r} is already loaded in this Python process; "
        f"start a new process before loading {profile_name!r}"
    )


def ensure_local_profile(
    *,
    profile_name: str = "aiida-renormalizer",
    email: str = "aiida-renormalizer@localhost",
    storage_path: str | Path | None = None,
    set_default: bool = True,
    load: bool = True,
) -> ProfileBootstrapReport:
    """Create or validate a service-free AiiDA profile for local provenance.

    The profile uses ``core.sqlite_dos`` and no broker. This is sufficient for
    direct calcfunction execution and ORM provenance, but not for daemon-backed
    process control or remote CalcJob execution.
    """
    profile_name = profile_name.strip()
    email = email.strip()
    if not profile_name:
        raise BootstrapConfigurationError("profile_name must be non-empty")
    if not email:
        raise BootstrapConfigurationError("email must be non-empty")

    config = get_config(create=True)
    explicit_storage_path = Path(storage_path).expanduser().resolve() if storage_path else None
    created = profile_name not in config.profile_names

    if load:
        _validate_profile_load_request(config, profile_name)

    if created:
        resolved_storage_path = explicit_storage_path or _default_storage_path(
            config.dirpath,
            profile_name,
        )
        if resolved_storage_path.is_file():
            raise BootstrapConfigurationError(
                f"storage path is a file, expected a directory: {resolved_storage_path}"
            )
        if resolved_storage_path.is_dir() and any(resolved_storage_path.iterdir()):
            raise BootstrapConfigurationError(
                f"new profile storage path must be empty: {resolved_storage_path}"
            )
        profile = create_profile(
            config,
            storage_backend=LOCAL_STORAGE_BACKEND,
            storage_config={"filepath": str(resolved_storage_path)},
            broker_backend=None,
            broker_config=None,
            name=profile_name,
            email=email,
        )
    else:
        profile = config.get_profile(profile_name)
        resolved_storage_path = _validate_existing_profile(profile, explicit_storage_path)

    if load:
        _load_profile_without_implicit_switch(profile)

    if set_default and config.default_profile_name != profile_name:
        config.set_default_profile(profile_name, overwrite=True)
        config.store()

    return ProfileBootstrapReport(
        profile_name=profile.name,
        profile_uuid=profile.uuid,
        created=created,
        config_dir=str(Path(config.dirpath).resolve()),
        storage_backend=profile.storage_backend,
        storage_path=str(resolved_storage_path),
        broker_backend=profile.process_control_backend,
        default_user_email=profile.default_user_email,
        default_profile=config.default_profile_name == profile_name,
        loaded=get_manager().get_profile() is not None
        and get_manager().get_profile().uuid == profile.uuid,
    )
