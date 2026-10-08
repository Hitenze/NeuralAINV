from dataclasses import asdict
import json
import pickle

import pytest

from neuralainv import training


def saved_run(directory, updates, outer):
    config = training.TrainingConfig.smoke()
    state = {"phase": 0, "outer": outer, "updates": updates}
    (directory / "config.json").write_text(json.dumps(asdict(config)))
    (directory / "resume.pkl").write_bytes(pickle.dumps(state))
    (directory / "history.csv").write_text("updates,stage\n2,1\n")
    return config, state


@pytest.mark.parametrize("updates,outer,target", [(2, 1, 2), (6, 3, 6), (6, 3, 2)])
def test_resume_at_stop_target_leaves_saved_run_unchanged(
    tmp_path, monkeypatch, updates, outer, target
):
    config, expected = saved_run(tmp_path, updates, outer)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def unexpected(*args, **kwargs):
        pytest.fail("an already reached target must not initialize or update")

    for name in ("initialize", "start_phase", "make_scan", "outer_update"):
        monkeypatch.setattr(training, name, unexpected)
    actual = training.train(
        config, tmp_path, resume=True, stop_after_updates=target
    )
    assert actual == expected
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_resume_below_stop_target_continues(tmp_path, monkeypatch):
    config, _ = saved_run(tmp_path, updates=2, outer=1)
    monkeypatch.setattr(training, "make_scan", lambda *args: None)

    class UpdateReached(Exception):
        pass

    def update(*args):
        raise UpdateReached

    monkeypatch.setattr(training, "outer_update", update)
    with pytest.raises(UpdateReached):
        training.train(config, tmp_path, resume=True, stop_after_updates=4)
