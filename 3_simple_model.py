# 3_simple_model.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train_cat = np.load('y_train_cat.npy')
y_val_cat = np.load('y_val_cat.npy')
y_test = np.load('y_test.npy')

# Reshape for MLP (flatten the images)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Flattened shapes:")
print(f"X_train_flat: {X_train_flat.shape}")
print(f"X_val_flat: {X_val_flat.shape}")
print(f"X_test_flat: {X_test_flat.shape}")

# Build MLP model
def create_mlp_model():
    model = Sequential([
        # Input layer (784 neurons for 28x28 images)
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        
        Dense(256, activation='relu'),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        # Output layer (10 classes)
        Dense(10, activation='softmax')
    ])
    
    return model

model = create_mlp_model()
model.summary()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train_flat, y_train_cat,
    batch_size=128,
    epochs=20,
    validation_data=(X_val_flat, y_val_cat),
    verbose=1
)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('mlp_training_history.png')
plt.show()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions
y_pred = model.predict(X_test_flat)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - MLP Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('mlp_confusion_matrix.png')
plt.show()

# Save model
model.save('mlp_digit_recognizer.h5')
print("\n✓ MLP model saved as 'mlp_digit_recognizer.h5'")