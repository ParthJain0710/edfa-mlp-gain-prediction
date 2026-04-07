import tensorflow as tf
from tensorflow.keras import layers, Model


def build_mlp(input_dim: int, output_dim: int, dropout_rate: float = 0.2) -> Model:
    """7-layer MLP: [512, 384, 256, 192, 128, 96, 64] with BatchNorm + Dropout."""
    layer_sizes = [512, 384, 256, 192, 128, 96, 64]

    inputs = tf.keras.Input(shape=(input_dim,), name="spectrum_input")
    x = inputs

    for i, size in enumerate(layer_sizes):
        x = layers.Dense(size, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i+1}")(x)

    outputs = layers.Dense(output_dim, name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name="EDFA_MLP")
