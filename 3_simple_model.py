# 3_simple_model.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load preprocessed data
print("Loading preprocessed data...")
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')

# Load ALL label files
y_train_cat = np.load('y_train_cat.npy')
y_val_cat = np.load('y_val_cat.npy')
y_test_cat = np.load('y_test_cat.npy')  # This line was missing!
y_test = np.load('y_test.npy')  # Original labels for confusion matrix

print("✅ All data loaded successfully!")
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape (one-hot): {y_test_cat.shape}")
print(f"Test labels shape (original): {y_test.shape}")

# Reshape for MLP (flatten the images)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"\nFlattened shapes:")
print(f"X_train_flat: {X_train_flat.shape}")
print(f"X_val_flat: {X_val_flat.shape}")
print(f"X_test_flat: {X_test_flat.shape}")

# Build MLP model
def create_mlp_model():
    model = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        
        Dense(256, activation='relu'),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
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
print("\n🔄 Training MLP model...")
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
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
ax1.set_title('Model Accuracy', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Train Loss', color='blue')
ax2.plot(history.history['val_loss'], label='Val Loss', color='orange')
ax2.set_title('Model Loss', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_training_history.png', dpi=300, bbox_inches='tight')
print("\n✅ Training history saved as 'mlp_training_history.png'")
plt.show(block=False)
plt.pause(2)
plt.close()

# Evaluate on test set
print("\n📊 Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test_cat, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✅ Test Loss: {test_loss:.4f}")

# Make predictions
print("\n🔍 Making predictions...")
y_pred = model.predict(X_test_flat, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - MLP Model', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved as 'mlp_confusion_matrix.png'")
plt.show(block=False)
plt.pause(2)
plt.close()

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\n📊 Per-class Accuracy:")
for i, acc in enumerate(class_accuracy):
    print(f"   Digit {i}: {acc:.4f} ({acc*100:.2f}%)")

# Save model
model.save('mlp_digit_recognizer.h5')
print("\n✅ MLP model saved as 'mlp_digit_recognizer.h5'")

# Also save in Keras v3 format
model.save('mlp_digit_recognizer.keras')
print("✅ MLP model saved as 'mlp_digit_recognizer.keras'")

print("\n" + "="*50)
print("🎉 MLP MODEL TRAINING COMPLETE!")
print("="*50)
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print("="*50)