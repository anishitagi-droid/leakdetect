import numpy as np
import pandas as pd
import pytest


def _csv(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    path = tmp_path / "d.csv"
    pd.DataFrame({"y": y, "x": rng.normal(size=300), "leak": y + rng.normal(scale=0.01, size=300)}).to_csv(
        path, index=False
    )
    return str(path)


def test_cli_flags_the_leak_and_exits_nonzero(tmp_path, capsys):
    from leakdetect.__main__ import main

    assert main([_csv(tmp_path), "--target", "y", "--positive", "1"]) == 1
    out = capsys.readouterr().out
    assert "leak" in out and "single-feature-predictive-power" in out


def test_cli_exit_status_is_configurable_and_json_is_valid(tmp_path, capsys):
    import json

    from leakdetect.__main__ import main

    assert main([_csv(tmp_path), "--target", "y", "--positive", "1", "--fail-severity", "0", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["max_severity"] >= 4


def test_cli_rejects_a_bad_positive_label_and_unknown_columns(tmp_path):
    from leakdetect.__main__ import main

    path = _csv(tmp_path)
    with pytest.raises(SystemExit, match="not a value"):
        main([path, "--target", "y", "--positive", "maybe"])
    with pytest.raises(SystemExit, match="not in"):
        main([path, "--target", "nope", "--positive", "1"])


@pytest.mark.parametrize(("cast", "positive"), [(float, "1"), (bool, "true"), (bool, "True")])
def test_cli_matches_positive_label_across_numeric_and_boolean_targets(tmp_path, capsys, cast, positive):
    from leakdetect.__main__ import main

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    path = tmp_path / "t.csv"
    pd.DataFrame({"y": y.astype(cast), "x": rng.normal(size=300), "leak": y + rng.normal(scale=0.01, size=300)}).to_csv(
        path, index=False
    )
    assert main([str(path), "--target", "y", "--positive", positive]) == 1
    assert "leak" in capsys.readouterr().out


def test_cli_group_flag_reports_entity_leakage(tmp_path, capsys):
    from leakdetect.__main__ import main

    rng = np.random.default_rng(0)
    subject = np.repeat(np.arange(60), 6)
    label = rng.permutation(np.arange(60) % 2)[subject]
    sig = rng.normal(size=(60, 5))[subject] + rng.normal(scale=0.3, size=(360, 5))
    path = tmp_path / "g.csv"
    pd.DataFrame(sig, columns=list("abcde")).assign(y=label, subject=subject).to_csv(path, index=False)
    assert main([str(path), "--target", "y", "--positive", "1", "--group", "subject"]) == 1
    assert "group-leakage" in capsys.readouterr().out
