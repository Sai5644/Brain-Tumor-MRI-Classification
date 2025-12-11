# train_brain_tumor.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from cnn_model import build_cnn_model

# ---- Setup ----
SAVE = True
SEED = 111
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_TYPES = ['pituitary', 'notumor', 'meningioma', 'glioma']
N_TYPES = len(CLASS_TYPES)

USER_PATH = os.getcwd()
train_dir = os.path.join(USER_PATH, "Training")
test_dir = os.path.join(USER_PATH, "Testing")

# ---- Data Generators ----
image_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   brightness_range=(0.85, 1.15),
                                   shear_range=12.5,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    seed=SEED)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed=SEED)

# ---- Model ----
image_shape = (image_size[0], image_size[1], 3)
model = build_cnn_model(image_shape=image_shape, n_classes=N_TYPES, seed=SEED)

model.summary()

# ---- Callbacks ----
es = EarlyStopping(monitor='loss', patience=8, verbose=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=True)

# ---- Train ----
epochs = 40
steps_per_epoch = train_generator.samples // batch_size
validation_steps = test_generator.samples // batch_size

history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=validation_steps,
                    callbacks=[es, rlr])

# ---- Evaluate ----
loss, acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"\nTest Accuracy: {acc:.4f}")

# ---- Save Model ----
model_path = os.path.join(OUTPUT_DIR, "brain_tumor_model.h5")
model.save(model_path)
print(f"✅ Model saved at {model_path}")

# ---- Plots ----
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_plots.png"))
plt.show()

# ---- Confusion Matrix ----
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()
