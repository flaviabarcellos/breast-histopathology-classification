"""Train using validation only; reserve the test split for a separate command."""
import json

import tensorflow as tf

from .data import make_dataset
from .model import build_model, learning_rate, set_trainable_blocks


def train(config, manifest, output):
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    train_data = make_dataset(manifest[manifest.split == "train"], config, training=True)
    validation = make_dataset(manifest[manifest.split == "validation"], config)
    model = build_model(config)
    scores = []
    for number, blocks in enumerate(config["stages"]):
        stage = output / f"stage_{number}"
        stage.mkdir()
        set_trainable_blocks(model, blocks)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3 if number == 0 else 1e-5),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(stage / "best.keras"),
                                               monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"],
                                            restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(str(stage / "history.csv")),
            tf.keras.callbacks.TensorBoard(log_dir=str(stage / "tensorboard"))
        ]
        if number:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(learning_rate))
        history = model.fit(train_data, validation_data=validation,
                            epochs=config["epochs_per_stage"], callbacks=callbacks)
        model = tf.keras.models.load_model(stage / "best.keras")
        scores.append((min(history.history["val_loss"]), stage / "best.keras"))
    _, best = min(scores, key=lambda item: item[0])
    tf.keras.models.load_model(best).save(output / "best.keras")
    (output / "selection.json").write_text(json.dumps(
        {"criterion": "minimum validation loss across stages", "selected": str(best)}, indent=2
    ) + "\n")
