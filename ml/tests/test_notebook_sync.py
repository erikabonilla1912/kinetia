"""El notebook y ml/ no pueden desincronizarse."""
from __future__ import annotations

import base64
import gzip
import json
import re
from pathlib import Path

import build_notebook as bn
from dataset import DEFAULT_MANIFEST, manifest_sha256

ML_DIR = Path(__file__).resolve().parents[1]


def _notebook():
    return json.loads(bn.NOTEBOOK.read_text(encoding="utf-8"))


def _sources(nb):
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def test_notebook_equals_fresh_build():
    assert _sources(_notebook()) == _sources(bn.build()), "Ejecuta: python ml/build_notebook.py"


def test_every_embedded_file_matches_ml_dir():
    cells = {}
    for kind, text in _sources(_notebook()):
        match = re.match(r"%%writefile ml/(\S+)\n", text) if kind == "code" else None
        if match:
            cells[match.group(1)] = text[match.end():]
    assert set(cells) == set(bn.EMBEDDED)
    for rel, text in cells.items():
        assert text == (ML_DIR / rel).read_bytes().decode("utf-8").replace("\r\n", "\n"), f"{rel} desincronizado"


def test_every_ml_python_file_is_embedded():
    expected = {p.name for p in ML_DIR.glob("*.py")} - {"build_notebook.py", "make_dummy_model.py"}  # este ultimo es del Agente Backend
    assert expected <= set(bn.EMBEDDED), f"Sin incrustar: {expected - set(bn.EMBEDDED)}"


def test_embedded_manifest_matches_file():
    text = next(t for _, t in _sources(_notebook()) if "_MANIFEST_B64" in t)
    digest = re.search(r'MANIFEST_SHA256 = "([0-9a-f]{64})"', text).group(1)
    assert digest == manifest_sha256(DEFAULT_MANIFEST)
    blob = re.search(r'_MANIFEST_B64 = """\n(.*?)\n"""', text, re.S).group(1)
    data = gzip.decompress(base64.b64decode(blob))
    assert data == DEFAULT_MANIFEST.read_bytes().replace(b"\r\n", b"\n")


def test_committed_notebook_has_no_outputs():
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_required_sections_are_present():
    text = "\n".join(t for _, t in _sources(_notebook()))
    for needle in ("SMOKE_TEST", "drive.mount", "NO HAY GPU", "kinetia_artifacts.zip", "history.json", "evaluate.py", "SOBREAJUSTE", "SUBAJUSTE", "train.py", "export.py"):
        assert needle in text, needle
