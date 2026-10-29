"""Notebook architecture with serializable preprocessing and explicit fine-tuning."""
from tensorflow import keras


def build_model(config):
    height, width = config["crop_size"]
    inputs = keras.Input((*config["image_size"], 3), name="image")
    # Explicit CenterCrop keeps inference shapes stable across Keras versions.
    x = keras.layers.RandomCrop(height, width, seed=config["seed"])(inputs)
    x = keras.layers.CenterCrop(height, width)(x)
    x = keras.layers.RandomFlip("horizontal_and_vertical", seed=config["seed"])(x)
    x = keras.layers.RandomRotation(20 / 360, seed=config["seed"])(x)
    x = keras.layers.RandomTranslation(0.2, 0.2, seed=config["seed"])(x)
    x = keras.layers.RandomZoom(0.2, seed=config["seed"])(x)
    backbone = keras.applications.EfficientNetV2B1(
        include_top=False, weights=config["weights"], input_shape=(height, width, 3),
        include_preprocessing=True
    )
    shallow = keras.Model(backbone.input, backbone.get_layer("block3c_add").output,
                          name="efficientnetv2b1_shallow")
    shallow.trainable = False
    x = shallow(x, training=False)
    branches = [keras.layers.Conv2D(filters, kernel, padding="same", activation="relu")(x)
                for filters, kernel in ((64, 1), (128, 3), (32, 5))]
    branches.append(keras.layers.MaxPooling2D(3, strides=1, padding="same")(x))
    x = keras.layers.Concatenate()(branches)
    x = keras.layers.BatchNormalization(name="inception_bn")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(2, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs)


def set_trainable_blocks(model, blocks):
    backbone = model.get_layer("efficientnetv2b1_shallow")
    backbone.trainable = bool(blocks)
    for layer in backbone.layers:
        layer.trainable = (any(layer.name.startswith(block) for block in blocks)
                           and not isinstance(layer, keras.layers.BatchNormalization))


def learning_rate(epoch, warmup=25, minimum=1e-5, maximum=5e-5, decay=0.8):
    if epoch < warmup:
        return minimum + (maximum - minimum) * epoch / warmup
    return minimum + (maximum - minimum) * decay ** (epoch - warmup)
