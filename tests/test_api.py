import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from ecg import config


@pytest.fixture(scope="module")
def client():
    if not config.MODEL_PATH.exists() and not config.LEGACY_MODEL_PATH.exists():
        pytest.skip("no trained model available - run scripts/train_model.py first")
    from app.main import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    # must never leak the server's filesystem layout
    assert "path" not in json_keys_lower(body)
    assert str(config.PROJECT_ROOT) not in str(body)


def test_model_info_has_no_filesystem_paths(client):
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "architecture" in body
    assert str(config.PROJECT_ROOT) not in str(body)


def json_keys_lower(d):
    return {k.lower() for k in d}


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"ECG" in resp.data


def test_predict_without_file(client):
    resp = client.post("/predict", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_rejects_wrong_extension(client):
    data = {"file": (io.BytesIO(b"not an ecg"), "malicious.py")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_rejects_oversized_upload(client):
    big = io.BytesIO(b"\x00" * (6 * 1024 * 1024))  # 6MB > the 5MB cap
    data = {"file": (big, "huge.dat")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 413
    assert "error" in resp.get_json()


def test_predict_rejects_oversized_filename(client):
    # An OS-level filename length limit (NTFS: 255 bytes/component) must
    # degrade to a clean 400, not an unhandled OSError -> bare 500.
    long_name = "a" * 300 + ".dat"
    data = {"file": (io.BytesIO(b"\x00" * 4800), long_name)}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_rejects_path_traversal_filename(client):
    # secure_filename() must strip any path components so this can never
    # write outside the per-request upload directory.
    data = {"file": (io.BytesIO(b"\x00" * 100), "../../evil.dat")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    # whatever the outcome, it must not be a server crash and must not
    # create a file outside the sandboxed uploads dir
    assert resp.status_code in (400, 422)
    assert not (config.PROJECT_ROOT.parent / "evil.dat").exists()
    assert not (config.PROJECT_ROOT / "evil.dat").exists()


def test_predict_on_bundled_demo_record(client):
    demo = config.DEMO_RECORD_PATH
    if not Path(str(demo) + ".dat").exists():
        pytest.skip("bundled demo record not present")

    with open(str(demo) + ".dat", "rb") as dat_f, open(str(demo) + ".hea", "rb") as hea_f:
        data = {
            "file": (io.BytesIO(dat_f.read()), "00001_hr.dat"),
            "hea_file": (io.BytesIO(hea_f.read()), "00001_hr.hea"),
        }
        resp = client.post("/predict", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] in ("normal", "abnormal")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["beats_analyzed"] > 0
    assert len(body["signal"]) > 0
    assert body["processing_time_ms"] > 0
    assert "architecture" in body["model"]
    assert body["explainability"] is not None
    assert len(body["explainability"]["saliency"]) == len(body["explainability"]["beat_window"])
    assert all(0.0 <= v <= 1.0 for v in body["explainability"]["saliency"])
    assert "not a clinical explanation" in body["explainability"]["note"]


def test_predict_confidence_uses_mean_probability_not_majority_vote(client, monkeypatch):
    # Regression test for a real bug found in this hardening pass:
    # /predict used to compute np.mean(predictions > 0.5) - the fraction
    # of beats individually voted abnormal - not the mean probability.
    # For beats that are all < 0.5 but with varying confidence, the old
    # code always reported exactly 100% confidence regardless of how
    # weak the signal actually was. Mock the model to return known,
    # mixed-but-same-direction probabilities and assert the response
    # reflects genuine mean-pooling + calibration, not that degenerate case.
    import app.main as main_module
    from ecg.postprocessing import aggregate, apply_temperature

    fixed_probs = np.array([0.1, 0.2, 0.3, 0.4])  # all < 0.5, mean = 0.25, not 0.0 or 1.0

    class _FakeModel:
        def predict(self, X, verbose=0):
            return fixed_probs.reshape(-1, 1)

    monkeypatch.setattr(main_module, "model", _FakeModel())

    demo = config.DEMO_RECORD_PATH
    if not Path(str(demo) + ".dat").exists():
        pytest.skip("bundled demo record not present")
    with open(str(demo) + ".dat", "rb") as dat_f, open(str(demo) + ".hea", "rb") as hea_f:
        data = {
            "file": (io.BytesIO(dat_f.read()), "00001_hr.dat"),
            "hea_file": (io.BytesIO(hea_f.read()), "00001_hr.hea"),
        }
        resp = client.post("/predict", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    body = resp.get_json()

    expected_record_prob = aggregate(fixed_probs, method=config.AGGREGATION_METHOD)
    expected_calibrated = float(apply_temperature(np.array([expected_record_prob]), config.CALIBRATION_TEMPERATURE)[0])
    expected_confidence = expected_calibrated if expected_calibrated > 0.5 else 1 - expected_calibrated

    assert body["prediction"] == "normal"  # mean prob 0.25 (pre-calibration) is well under 0.5
    assert body["confidence"] == pytest.approx(round(expected_confidence, 4), abs=1e-3)
    # the old buggy formula would have produced exactly 1.0 here (100%)
    assert body["confidence"] != pytest.approx(1.0, abs=1e-6)


def test_predict_rejects_wrong_hea_extension(client):
    data = {
        "file": (io.BytesIO(b"\x00" * 4800), "record.dat"),
        "hea_file": (io.BytesIO(b"not a header"), "record.txt"),
    }
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_handles_flat_headerless_signal_gracefully(client):
    # No .hea -> raw fallback path; an all-zero signal has no R-peaks at
    # all, so this must be rejected cleanly (422) rather than crash or
    # silently return a prediction with no real evidence behind it.
    silence = (b"\x00\x00" * 12) * 2000  # 2000 samples x 12 channels, all zero
    data = {"file": (io.BytesIO(silence), "silence.dat")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 422
    assert "error" in resp.get_json()


def test_predict_handles_malformed_wfdb_record_gracefully(client):
    # A real .hea paired with garbage/undersized .dat content: wfdb will
    # fail to parse it. Must degrade to a clean 422, never a 500 with a
    # leaked stack trace or filesystem path.
    demo = config.DEMO_RECORD_PATH
    if not Path(str(demo) + ".hea").exists():
        pytest.skip("bundled demo record not present")
    with open(str(demo) + ".hea", "rb") as hea_f:
        hea_bytes = hea_f.read()
    data = {
        "file": (io.BytesIO(b"\x01\x02\x03"), "00001_hr.dat"),
        "hea_file": (io.BytesIO(hea_bytes), "00001_hr.hea"),
    }
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 422
    body = resp.get_json()
    assert "error" in body
    assert str(config.PROJECT_ROOT) not in str(body)
    assert "Traceback" not in str(body)
