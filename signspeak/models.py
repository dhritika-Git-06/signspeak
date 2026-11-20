"""Keras CNN and LSTM builders. Architectures are mirrored in the browser."""

from __future__ import annotations

import numpy as np


def build_static(num_classes: int):
    import keras

    inp = keras.Input(shape=(21, 3), name="landmarks")
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1")(inp)
    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.GlobalAveragePooling1D(name="pool")(x)
    emb = keras.layers.Dense(128, activation="relu", name="embed")(x)
    x = keras.layers.Dropout(0.3, name="drop")(emb)
    out = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inp, out, name="static_cnn")
    embed_model = keras.Model(inp, emb, name="static_embed")
    return model, embed_model


def build_dynamic(num_classes: int, timesteps: int = 30, features: int = 63):
    import keras

    inp = keras.Input(shape=(timesteps, features), name="sequence")
    x = keras.layers.LSTM(64, return_sequences=True, use_cudnn=False, name="lstm1")(inp)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.LSTM(32, use_cudnn=False, name="lstm2")(x)
    x = keras.layers.Dropout(0.25)(x)
    emb = keras.layers.Dense(32, activation="relu", name="embed")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="probs")(emb)
    model = keras.Model(inp, out, name="dynamic_lstm")
    embed_model = keras.Model(inp, emb, name="dynamic_embed")
    return model, embed_model


def static_weight_dict(model) -> dict[str, np.ndarray]:
    names = ["conv1", "conv2", "conv3", "embed", "probs"]
    out = {}
    for name in names:
        kernel, bias = model.get_layer(name).get_weights()
        out[f"{name}_k"] = np.asarray(kernel, dtype=np.float32)
        out[f"{name}_b"] = np.asarray(bias, dtype=np.float32)
    return out


def dynamic_weight_dict(model) -> dict[str, np.ndarray]:
    out = {}
    for name in ("lstm1", "lstm2"):
        kernel, recurrent, bias = model.get_layer(name).get_weights()
        out[f"{name}_k"] = np.asarray(kernel, dtype=np.float32)
        out[f"{name}_r"] = np.asarray(recurrent, dtype=np.float32)
        out[f"{name}_b"] = np.asarray(bias, dtype=np.float32)
    for name in ("embed", "probs"):
        kernel, bias = model.get_layer(name).get_weights()
        out[f"{name}_k"] = np.asarray(kernel, dtype=np.float32)
        out[f"{name}_b"] = np.asarray(bias, dtype=np.float32)
    return out


STATIC_ORDER = [
    "conv1_k",
    "conv1_b",
    "conv2_k",
    "conv2_b",
    "conv3_k",
    "conv3_b",
    "embed_k",
    "embed_b",
    "probs_k",
    "probs_b",
]

DYNAMIC_ORDER = [
    "lstm1_k",
    "lstm1_r",
    "lstm1_b",
    "lstm2_k",
    "lstm2_r",
    "lstm2_b",
    "embed_k",
    "embed_b",
    "probs_k",
    "probs_b",
]
