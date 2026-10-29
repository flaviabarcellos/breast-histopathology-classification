import pandas as pd
import pytest

from breakhis.data import index_images, split_patients


def test_index_uses_native_patient_and_diagnosis(tmp_path):
    (tmp_path / "SOB_B_A-14-22549AB-40-001.png").touch()
    (tmp_path / "SOB_M_DC-14-2985-100-001.png").touch()
    frame = index_images(tmp_path, "40")
    assert frame.patient_id.tolist() == ["14-22549AB"]
    assert frame.label.tolist() == [0]


def test_unknown_names_fail_instead_of_guessing(tmp_path):
    (tmp_path / "unknown.png").touch()
    with pytest.raises(ValueError, match="patient identity"):
        index_images(tmp_path, "40")


def test_patient_splits_are_disjoint_stratified_and_reproducible():
    rows = [{"path": f"{label}_{patient}_{image}.png",
             "patient_id": f"{label}_{patient}", "label": label}
            for label in (0, 1) for patient in range(30) for image in range(3)]
    frame = pd.DataFrame(rows)
    result = split_patients(frame)
    pd.testing.assert_frame_equal(result, split_patients(frame))
    assert len(result) == len(frame)
    assert result.path.nunique() == len(frame)
    assert result.groupby("patient_id").split.nunique().eq(1).all()
    assert result.groupby("split").label.nunique().eq(2).all()
    assert set(result.split) == {"train", "validation", "test"}


def test_invalid_fractions_fail():
    with pytest.raises(ValueError, match="sum to one"):
        split_patients(pd.DataFrame(), train=0.8)
