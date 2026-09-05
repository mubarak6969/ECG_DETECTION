import numpy as np

from ecg.multilabel import SUPERCLASSES, _target_vector

CODE_TO_CLASS = {"NORM": "NORM", "IMI": "MI", "NDT": "STTC", "LAFB": "CD", "LVH": "HYP", "SR": None}
# SR (sinus rhythm) intentionally has no mapped class - it's a rhythm
# statement, not one of the 5 diagnostic superclasses, matching the real
# scp_statements.csv where many codes have diagnostic_class = NaN.
CODE_TO_CLASS = {k: v for k, v in CODE_TO_CLASS.items() if v is not None}


def test_target_vector_single_class():
    vec = _target_vector("{'NORM': 100.0}", CODE_TO_CLASS)
    expected = np.zeros(len(SUPERCLASSES))
    expected[SUPERCLASSES.index("NORM")] = 1.0
    assert np.array_equal(vec, expected)


def test_target_vector_ignores_zero_likelihood_codes():
    vec = _target_vector("{'NORM': 100.0, 'IMI': 0.0}", CODE_TO_CLASS)
    expected = np.zeros(len(SUPERCLASSES))
    expected[SUPERCLASSES.index("NORM")] = 1.0
    assert np.array_equal(vec, expected)  # IMI at 0% doesn't count


def test_target_vector_true_multilabel():
    vec = _target_vector("{'IMI': 80.0, 'LAFB': 60.0}", CODE_TO_CLASS)
    assert vec[SUPERCLASSES.index("MI")] == 1.0
    assert vec[SUPERCLASSES.index("CD")] == 1.0
    assert vec.sum() == 2.0


def test_target_vector_unmappable_code_returns_none():
    # SR has no diagnostic_class in scp_statements.csv (it's a rhythm
    # statement) - a record with only unmappable codes should be dropped
    # by load_multilabel_metadata, signaled here by returning None.
    assert _target_vector("{'SR': 100.0}", CODE_TO_CLASS) is None


def test_target_vector_malformed_string_returns_none():
    assert _target_vector("not a dict", CODE_TO_CLASS) is None


def test_superclasses_match_ptbxl_diagnostic_classes():
    assert set(SUPERCLASSES) == {"NORM", "MI", "STTC", "CD", "HYP"}
