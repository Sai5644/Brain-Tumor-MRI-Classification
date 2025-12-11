# cnn_model.py
# --------------------------
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_model(image_shape, n_classes, seed=111):
    """
    Builds and compiles a CNN model for brain tumor classification.
    """
    tf.keras.utils.set_random_seed(seed)

    model = models.Sequential([
        Conv2D(32, (4, 4), activation="relu", input_shape=image_shape),
        MaxPooling2D(pool_size=(3, 3)),

        Conv2D(64, (4, 4), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),

        Conv2D(128, (4, 4), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),

        Conv2D(128, (4, 4), activation="relu"),
        Flatten(),

        Dense(512, activation="relu"),
        Dropout(0.5, seed=seed),
        Dense(n_classes, activation="softmax")
    ])

    optimizer = Adam(learning_rate=0.001, beta_1=0.869, beta_2=0.995)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
