# 1_data_exploration.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print basic information
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Pixel values range: {X_train.min()} to {X_train.max()}")
print(f"Number of classes: {len(np.unique(y_train))}")
print(f"Classes: {np.unique(y_train)}")

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    # Find first occurrence of each digit
    idx = np.where(y_train == i)[0][0]
    axes[i].imshow(X_train[idx], cmap='gray')
    axes[i].set_title(f'Digit: {i}')
    axes[i].axis('off')

plt.suptitle('Sample Images from MNIST Dataset')
plt.tight_layout()
plt.savefig('sample_digits.png')
plt.show()

# Check class distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Training set distribution
train_counts = np.bincount(y_train)
ax1.bar(range(10), train_counts)
ax1.set_title('Training Set Class Distribution')
ax1.set_xlabel('Digit')
ax1.set_ylabel('Count')

# Test set distribution
test_counts = np.bincount(y_test)
ax2.bar(range(10), test_counts)
ax2.set_title('Test Set Class Distribution')
ax2.set_xlabel('Digit')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show(block=False)
plt.pause(2)  
plt.close()

print("\n✓ Data exploration complete!")