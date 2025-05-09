# Step 1: Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 2: Load the Boston Housing dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.boston_housing.load_data()

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# Step 3: Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build the linear regression DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # Linear output for regression
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

# Step 6: Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=8,
                    validation_split=0.1,
                    verbose=1)

# Step 7: Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print("Test Mean Absolute Error:", mae)
print("Test Mean Squared Error:", loss)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Housing Prices')
plt.xlabel('Actual Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.grid(True)
plt.show()
