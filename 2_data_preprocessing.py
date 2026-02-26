# 2_data_preprocessing.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load data
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print("Original shapes:")
print(f"X_train_full: {X_train_full.shape}")
print(f"X_test: {X_test.shape}")

# 1. Normalize pixel values (scale to 0-1)
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"\nAfter normalization - min: {X_train_full.min()}, max: {X_train_full.max()}")

# 2. Reshape for neural network (add channel dimension)
# CNNs expect shape: (samples, height, width, channels)
X_train_full = X_train_full.reshape(X_train_full.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(f"\nAfter reshaping for CNN: {X_train_full.shape}")

# 3. Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\nAfter split:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# 4. Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_val_cat = to_categorical(y_val, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"\nLabels shape after one-hot encoding:")
print(f"y_train_cat: {y_train_cat.shape}")
print(f"Sample: Digit 5 becomes {y_train_cat[0]}")

# 5. Data augmentation demonstration
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Show augmented samples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
sample_img = X_train[0].reshape(1, 28, 28, 1)

# Original image
axes[0, 0].imshow(sample_img.reshape(28, 28), cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Augmented versions
for i in range(1, 5):
    augmented = datagen.flow(sample_img, batch_size=1)[0]
    axes[0, i].imshow(augmented.reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f'Augmented {i}')
    axes[0, i].axis('off')

# More augmented samples
for i in range(5):
    augmented = datagen.flow(sample_img, batch_size=1)[0]
    axes[1, i].imshow(augmented.reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f'Augmented {i+5}')
    axes[1, i].axis('off')

plt.suptitle('Data Augmentation Examples')
plt.tight_layout()
plt.savefig('augmentation.png')
plt.show()

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)
np.save('y_train_cat.npy', y_train_cat)
np.save('y_val_cat.npy', y_val_cat)
np.save('y_test_cat.npy', y_test_cat)

print("\n✓ Preprocessed data saved!")