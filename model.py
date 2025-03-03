import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

# Load and clean data
df = pd.read_csv('churnpractice/Customer-Churn-Records.csv')

# 1. Remove unnecessary columns and the leaking 'Complain' column
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Complain']
df = df.drop(columns=columns_to_drop)

# Separate features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create scalers
robust_scaler = RobustScaler()
minmax_scaler = MinMaxScaler()
minmax_scaler_credit = MinMaxScaler()
minmax_scaler_balance = MinMaxScaler()
minmax_scaler_card = MinMaxScaler()

def preprocess_data(data, is_training=True):
    data = data.copy()
    
    # Add new feature interactions
    data['Credit_to_Balance_Ratio'] = data['CreditScore'] / (data['Balance'] + 1)
    data['Products_to_Tenure_Ratio'] = data['NumOfProducts'] / (data['Tenure'] + 1)
    
    # Create interaction features
    data['Balance_Per_Salary'] = data['Balance'] / (data['EstimatedSalary'] + 1)
    data['Points_Per_Product'] = data['Point Earned'] / (data['NumOfProducts'] + 1)
    data['Tenure_Age_Ratio'] = data['Tenure'] / (data['Age'] + 1)
    
    # Create polynomial features
    data['Age_Squared'] = np.square(data['Age'])
    data['Balance_Squared'] = np.square(data['Balance'])
    
    # Create binary flags
    data['Has_High_Balance'] = (data['Balance'] > data['Balance'].mean()).astype(int)
    data['Is_Senior'] = (data['Age'] > 60).astype(int)
    data['Is_Long_Tenure'] = (data['Tenure'] > 5).astype(int)
    
    # CreditScore - robust scaling then min max scaling
    if is_training:
        data['CreditScore'] = robust_scaler.fit_transform(data[['CreditScore']])
        data['CreditScore'] = minmax_scaler_credit.fit_transform(data[['CreditScore']])
    else:
        data['CreditScore'] = robust_scaler.transform(data[['CreditScore']])
        data['CreditScore'] = minmax_scaler_credit.transform(data[['CreditScore']])
    
    # Geography - one hot encoding
    data = pd.get_dummies(data, columns=['Geography'])
    
    # Gender - binary encoding
    data['Gender'] = (data['Gender'] == 'Male').astype(int)
    
    # Min/Max scaling for numeric columns
    numeric_columns = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 
                      'Satisfaction Score', 'Point Earned', 'Balance_Per_Salary',
                      'Points_Per_Product', 'Tenure_Age_Ratio', 'Age_Squared',
                      'Balance_Squared']
    
    if is_training:
        data[numeric_columns] = minmax_scaler.fit_transform(data[numeric_columns])
    else:
        data[numeric_columns] = minmax_scaler.transform(data[numeric_columns])
    
    # Card Type - ordinal encode then min/max scaling
    card_type_mapping = {
        'SILVER': 1,
        'GOLD': 2,
        'DIAMOND': 3,
        'PLATINUM': 4
    }
    data['Card Type'] = data['Card Type'].map(card_type_mapping)
    if is_training:
        data['Card Type'] = minmax_scaler_card.fit_transform(data[['Card Type']])
    else:
        data['Card Type'] = minmax_scaler_card.transform(data[['Card Type']])
    
    return data

# Apply preprocessing
X_train_processed = preprocess_data(X_train, is_training=True)
X_test_processed = preprocess_data(X_test, is_training=False)

# Define custom loss function
def custom_loss(y_true, y_pred):
    # Standard binary crossentropy
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    
    # Reduced penalty for missing churners (from 2.0 to 1.2)
    churner_penalty = 1.2 * tf.keras.backend.mean(y_true * (1 - y_pred))
    
    return bce + churner_penalty

# Keep undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_processed, y_train)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

# Custom callback for epoch printing
class EpochPrintCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch {epoch + 1}')
            print(f'Loss: {logs["loss"]:.4f}')
            print(f'Accuracy: {logs["accuracy"]:.4f}')

# Add learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.0001
)

# Model with gradient clipping
model = models.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(64, activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                kernel_constraint=tf.keras.constraints.MaxNorm(3)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                kernel_constraint=tf.keras.constraints.MaxNorm(3)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Compile with gradient clipping
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0  # Gradient clipping
    ),
    loss=custom_loss,
    metrics=['accuracy']
)

# Train the model with balanced data
history = model.fit(
    X_train_balanced, y_train_balanced,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EpochPrintCallback(), early_stopping, reduce_lr],
    verbose=0
)

# Evaluate and plot results
y_pred_proba = model.predict(X_test_processed)

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold that maximizes F1 score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nOptimal threshold: {optimal_threshold:.3f}")

# Use optimal threshold for predictions
y_pred_classes = (y_pred_proba > optimal_threshold).astype(int)

# Print updated classification report
print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test, y_pred_classes))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()