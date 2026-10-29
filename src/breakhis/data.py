"""Index native BreaKHis filenames and split patients before loading images."""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PATTERN = re.compile(
    r"^SOB_([BM])_[A-Z]+-(\d+-\w+)-(40|100|200|400)-\d+\.png$", re.IGNORECASE
)
CLASS_NAMES = ["benign", "malignant"]


def index_images(root, magnification):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Cannot extract patient identity from filename: {path}")
        diagnosis, patient, zoom = match.groups()
        if zoom == str(magnification).removesuffix("X"):
            rows.append({"path": str(path.resolve()), "patient_id": patient,
                         "label": int(diagnosis.upper() == "M"), "magnification": zoom})
    if not rows:
        raise ValueError("No matching BreaKHis PNG images found. Check data_dir and magnification.")
    frame = pd.DataFrame(rows)
    if frame.groupby("patient_id").label.nunique().max() != 1:
        raise ValueError("A patient has inconsistent diagnoses.")
    if frame.path.duplicated().any():
        raise ValueError("Duplicate image paths found.")
    return frame


def split_patients(frame, train=0.7, validation=0.1, test=0.2, seed=42):
    if min(train, validation, test) <= 0 or abs(train + validation + test - 1) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to one.")
    patients = frame[["patient_id", "label"]].drop_duplicates()
    try:
        train_ids, remaining = train_test_split(
            patients, train_size=train, random_state=seed, stratify=patients.label
        )
        val_ids, test_ids = train_test_split(
            remaining, test_size=test / (validation + test), random_state=seed,
            stratify=remaining.label
        )
    except ValueError as error:
        raise ValueError("Not enough patients per class for stratified patient splits.") from error
    parts = []
    for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        part = frame[frame.patient_id.isin(ids.patient_id)].copy()
        part["split"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def make_dataset(frame, config, training=False):
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((frame.path.tolist(), frame.label.tolist()))
    if training:
        dataset = dataset.shuffle(len(frame), seed=config["seed"], reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, config["image_size"])
        # Keep the [0, 255] range: EfficientNetV2 includes its own rescaling.
        return image, tf.one_hot(label, 2)

    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
        config["batch_size"]
    ).prefetch(tf.data.AUTOTUNE)
