import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "example_common", Path(__file__).resolve().parents[1] / "examples/_common.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


@pytest.mark.parametrize("filename", ["config.json", "rows.jsonl", "summary.json"])
def test_existing_results_are_preserved(tmp_path, monkeypatch, filename):
    artifact = tmp_path / filename
    artifact.write_text("existing record\n")

    def unexpected_load(path):
        pytest.fail("output check must precede weight loading")

    monkeypatch.setattr(common, "load_weights", unexpected_load)
    args = SimpleNamespace(output=tmp_path, weights="unused")
    with pytest.raises(FileExistsError, match="not empty"):
        common.prepare(args, "dense")
    assert artifact.read_text() == "existing record\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [filename]


@pytest.mark.parametrize("already_exists", [False, True])
def test_empty_output_is_allowed(tmp_path, monkeypatch, already_exists):
    output = tmp_path / "result"
    if already_exists:
        output.mkdir()
    kernel = object()
    monkeypatch.setattr(common, "load_weights", lambda path: kernel)
    args = SimpleNamespace(output=output, weights="unused")
    loaded, directory = common.prepare(args, "dense")
    assert loaded is kernel and directory == output
    assert (output / "config.json").exists()
