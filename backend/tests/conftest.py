from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
for entry in (str(BACKEND_DIR), str(BACKEND_DIR / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from api_fakes import FakeEngine, FakeFrameEncoder, make_settings, running_client  # noqa: E402


def generate_dummy(out: Path, *extra: str) -> Path:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "ml" / "make_dummy_model.py"), "--out", str(out), *extra],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def dummy_models(tmp_path_factory) -> Path:
    return generate_dummy(tmp_path_factory.mktemp("models"))


@pytest.fixture
def settings(tmp_path, dummy_models):
    return make_settings(tmp_path, dummy_models)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(settings, engine):
    with running_client(settings, lambda: engine) as test_client:
        yield test_client


@pytest.fixture
def fake_encoder(monkeypatch):
    """Sustituye a FFmpeg en las pruebas del ciclo de jobs que no prueban la codificación."""
    from app.services import encoder

    monkeypatch.setattr(encoder, "FrameEncoder", FakeFrameEncoder)
    monkeypatch.setattr(encoder, "find_tool", lambda name: name)
