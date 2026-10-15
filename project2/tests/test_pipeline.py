import json

import numpy as np
import pandas as pd
import pytest

from src.analyze import analyze, validate_raw
from src.simulation import load_config, run_study


def small_config():
    cfg = load_config("config/smoke.json")
    cfg["replicates"] = 4
    return cfg


def test_pipeline_reproduces_exactly_and_preserves_pairing(tmp_path):
    cfg = small_config()
    raw = run_study(cfg)
    pd.testing.assert_frame_equal(raw, run_study(cfg), check_exact=True)
    reordered = dict(cfg, gammas=list(reversed(cfg["gammas"])), replicates=5)
    extended = run_study(reordered)
    keys = ["mode", "pi0", "gamma", "replicate", "method"]
    pd.testing.assert_frame_equal(raw.sort_values(keys).reset_index(drop=True),
        extended[extended.replicate < 4].sort_values(keys).reset_index(drop=True), check_exact=True)
    analyze(raw, cfg, tmp_path, "results/reference/table2.csv")
    assert len(pd.read_csv(tmp_path / "design_matrix.csv")) == 16
    assert (tmp_path / "table2_comparison.csv").exists()
    fixed_alpha = raw[raw.method == "bh_fixed"]
    assert (fixed_alpha.groupby(["mode", "pi0", "replicate"]).rejections.nunique() == 1).all()
    original = raw[raw.method == "storey_fixed"]
    # Power remains undefined in the all-null condition, never silently set to zero.
    assert original[original.pi0 == 1].power.isna().all()


def test_raw_validation_rejects_corruption():
    cfg = small_config()
    raw = run_study(cfg)
    for corrupt in (raw.iloc[:-1], pd.concat([raw, raw.iloc[:1]])):
        with pytest.raises(ValueError):
            validate_raw(corrupt, cfg)
    raw.loc[0, "fdp"] = 1.1
    with pytest.raises(ValueError):
        validate_raw(raw, cfg)


def test_config_rejects_colliding_seed_keys(tmp_path):
    cfg = small_config()
    cfg["pi0_values"] = [0.50001]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="resolution"):
        load_config(path)


def test_simulated_power_and_fdr_against_known_truth():
    # Aggregate stochastic validation with generous uncertainty-based bounds.
    cfg = small_config()
    cfg.update(replicates=1000, pi0_values=[0.5], modes=["fixed"], boundary_pi0=[])
    raw = run_study(cfg)
    from src.metrics import analytic_targets
    for gamma in cfg["gammas"]:
        x = raw[(raw.method == "storey_fixed") & (raw.gamma == gamma)]
        truth = analytic_targets(cfg["m"], 0.5, gamma, 2, "fixed")
        for metric, target in (("power", "analytic_power"), ("fdp", "analytic_fdr")):
            se = x[metric].std(ddof=1) / np.sqrt(len(x))
            assert abs(x[metric].mean()-truth[target]) < 6*se


def test_cache_reuse_and_corruption_recovery(tmp_path, monkeypatch, caplog):
    from src.simulation import main
    import logging
    cfgpath = tmp_path / "config.json"
    cfgpath.write_text(json.dumps(small_config()))
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["simulation", "--config", str(cfgpath), "--output", str(output)])
    main()
    rawpath = output / "raw/replicates.csv.gz"
    first = rawpath.read_bytes()
    timestamp = rawpath.stat().st_mtime_ns
    with caplog.at_level(logging.INFO):
        main()
    assert "Reusing verified raw cache" in caplog.text
    assert rawpath.stat().st_mtime_ns == timestamp
    rawpath.write_bytes(b"corrupted")
    main()
    assert rawpath.read_bytes() == first
