from ecg import config


def test_fold_assignments_are_disjoint():
    train = set(config.TRAIN_FOLDS)
    assert config.VAL_FOLD not in train
    assert config.TEST_FOLD not in train
    assert config.VAL_FOLD != config.TEST_FOLD


def test_window_and_sampling_rate_are_positive():
    assert config.WINDOW_SIZE > 0
    assert config.SAMPLING_RATE_HZ > 0


def test_artifact_directories_exist():
    for d in (config.ARTIFACTS_DIR, config.SPLITS_DIR, config.MODELS_DIR, config.REPORTS_DIR):
        assert d.exists() and d.is_dir()


def test_upload_extension_allowlist_is_restrictive():
    assert config.ALLOWED_UPLOAD_EXTENSIONS == {".dat", ".hea"}


def test_dataset_root_is_configurable_via_env(monkeypatch):
    # config.py must read ECG_DATASET_ROOT rather than hardcoding a path -
    # verified by re-importing with the env var set.
    import importlib

    monkeypatch.setenv("ECG_DATASET_ROOT", "/tmp/some-other-ptbxl-copy")
    import ecg.config as cfg
    importlib.reload(cfg)
    try:
        assert str(cfg.DATASET_ROOT) in ("/tmp/some-other-ptbxl-copy", "\\tmp\\some-other-ptbxl-copy")
    finally:
        importlib.reload(cfg)  # restore real config for any later tests
