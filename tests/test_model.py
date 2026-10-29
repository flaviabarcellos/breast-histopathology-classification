import numpy as np
import tensorflow as tf

from breakhis.model import build_model, set_trainable_blocks


def test_rectangular_input_inference_and_serialization(tmp_path):
    config = {"crop_size": [64, 64], "image_size": [64, 96], "seed": 42, "weights": None}
    model = build_model(config)
    inputs = tf.zeros([1, 64, 96, 3])
    probabilities = model(inputs, training=False).numpy()
    assert probabilities.shape == (1, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    set_trainable_blocks(model, ["block3"])
    assert model.get_layer("efficientnetv2b1_shallow").trainable_weights
    path = tmp_path / "model.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored(inputs, training=False).numpy(), probabilities, atol=1e-6)
