import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from shutil import copy2
import matplotlib.pyplot as plt

# Set dataset directory
base_dir = "lungs"
train_dir = "lungs_split/train"
val_dir = "lungs_split/val"
test_dir = "lungs_split/test"

# Create train, val, and test directories if they don't exist
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(f"{folder}/NORMAL", exist_ok=True)
    os.makedirs(f"{folder}/PNEUMONIA", exist_ok=True)

# Split dataset into train, val, and test (70%, 15%, 15%)
def split_data(source, train_dest, val_dest, test_dest, split_ratios=(0.7, 0.15, 0.15)):
    all_files = os.listdir(source)
    np.random.shuffle(all_files)
    total_files = len(all_files)

    train_count = int(total_files * split_ratios[0])
    val_count = int(total_files * split_ratios[1])

    for i, file in enumerate(all_files):
        src_path = os.path.join(source, file)
        if i < train_count:
            dest_dir = train_dest
        elif i < train_count + val_count:
            dest_dir = val_dest
        else:
            dest_dir = test_dest
        copy2(src_path, dest_dir)

split_data(f"{base_dir}/NORMAL", f"{train_dir}/NORMAL", f"{val_dir}/NORMAL", f"{test_dir}/NORMAL")
split_data(f"{base_dir}/PNEUMONIA", f"{train_dir}/PNEUMONIA", f"{val_dir}/PNEUMONIA", f"{test_dir}/PNEUMONIA")

# Image data generators
datagen_train = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.2, horizontal_flip=True)
datagen_val_test = ImageDataGenerator(rescale=1./255)

train_data = datagen_train.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)
val_data = datagen_val_test.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)
test_data = datagen_val_test.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='binary', shuffle=False
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data
)

# Save the model
model.save("pneumonia_detection_model_v2.keras")
print("Model saved to pneumonia_detection_model_v2.keras")

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.2f}")

# Classification report
predictions = (model.predict(test_data) > 0.5).astype("int32")
true_labels = test_data.classes
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=["NORMAL", "PNEUMONIA"]))

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()
