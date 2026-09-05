import pandas as pd
import pytest

from ecg import config
from ecg.dataset import balance_train, class_distribution, classify_label, fold_split


@pytest.mark.parametrize("scp_codes,expected", [
    ("{'NORM': 100.0}", 0),
    # a 0.0-likelihood co-listed code means that finding was ruled out,
    # not that it's present - this record is still normal.
    ("{'NORM': 100.0, 'SBRAD': 0.0}", 0),
    ("{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}", 0),
    # but a genuinely nonzero co-finding makes it abnormal.
    ("{'NORM': 80.0, 'SBRAD': 20.0}", 1),
    ("{'MI': 80.0}", 1),
    ("not a dict", 1),
    ("", 1),
])
def test_classify_label(scp_codes, expected):
    assert classify_label(scp_codes) == expected


def _synthetic_metadata(n_patients=20, folds=10):
    rows = []
    ecg_id = 1
    for patient_id in range(n_patients):
        fold = (patient_id % folds) + 1
        for _ in range(2):  # two ECGs per patient, same fold - mirrors real PTB-XL
            label_scp = "{'NORM': 100.0}" if ecg_id % 10 == 0 else "{'MI': 80.0}"
            rows.append({
                "ecg_id": ecg_id, "patient_id": patient_id, "strat_fold": fold,
                "scp_codes": label_scp, "filename_lr": f"records100/0/{ecg_id}_lr",
            })
            ecg_id += 1
    df = pd.DataFrame(rows)
    df["label"] = df["scp_codes"].apply(classify_label)
    return df


def test_fold_split_has_no_patient_overlap():
    df = _synthetic_metadata()
    train_df, val_df, test_df = fold_split(df)

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert not (train_patients & val_patients)
    assert not (train_patients & test_patients)
    assert not (val_patients & test_patients)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_balance_train_keeps_all_normals_and_caps_abnormal_ratio():
    df = _synthetic_metadata(n_patients=40)
    train_df, _, _ = fold_split(df)
    balanced = balance_train(train_df, seed=0)

    n_normal = (balanced["label"] == 0).sum()
    n_abnormal = (balanced["label"] == 1).sum()
    assert n_normal == (train_df["label"] == 0).sum()
    assert n_abnormal <= n_normal * config.TRAIN_ABNORMAL_PER_NORMAL


def test_class_distribution():
    import numpy as np
    y = np.array([0, 0, 1, 1, 1])
    assert class_distribution(y) == {0: 2, 1: 3}
