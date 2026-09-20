"""Guards against work happening at import time.

Importing any module must be free and must never fail because of a missing
environment variable. A module-level `app = create_app()` once broke test
collection on a clean machine: the import read configuration, found no API
key, and raised before a single test ran. This test locks that door.
"""

import subprocess
import sys


def _import_in_clean_process(module: str) -> subprocess.CompletedProcess[str]:
    """Import in a subprocess with the service's env vars stripped, so the
    ambient shell cannot mask an import-time dependency."""
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )


def test_api_module_imports_without_any_configuration() -> None:
    result = _import_in_clean_process("intent_service.api.main")

    assert result.returncode == 0, f"import failed without config:\n{result.stderr}"


def test_adapter_module_imports_without_the_model_artifact() -> None:
    """Importing the adapter must not load a 700 KB joblib file."""
    result = _import_in_clean_process("intent_service.adapters.sklearn_model")

    assert result.returncode == 0, f"import failed:\n{result.stderr}"


def test_no_module_level_app_instance() -> None:
    """`app` must be built by an explicit call, never on import."""
    import intent_service.api.main as main_module

    assert not hasattr(main_module, "app"), (
        "main.py exposes a module-level `app`; use the create_app factory instead"
    )
