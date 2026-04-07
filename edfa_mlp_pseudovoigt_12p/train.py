import os
import tensorflow as tf


def train_model(model, train_X, train_y, val_X, val_y, config):
    """Compile and fit the model; returns Keras History object."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    model_path = os.path.join(config.MODEL_DIR, f"{config.DATASET_NAME}_best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.PATIENCE // 2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Best model saved to: {model_path}")
    return history
