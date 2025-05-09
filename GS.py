# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# Step 3: Load the CSV from the downloaded path
data = pd.read_csv(r"Goog.csv")  # Adjust filename if needed
print(data.head())

# Step 4: Use only the 'Close' prices for prediction
close_prices = data['Close'].values.reshape(-1, 1)

# Step 5: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Step 6: Create time series sequences
X, y = [], []
time_steps = 60
for i in range(time_steps, len(scaled_prices)):
    X.append(scaled_prices[i - time_steps:i])
    y.append(scaled_prices[i])

X = np.array(X)
y = np.array(y)

# Step 7: Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 8: Build the RNN model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

# Step 9: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 10: Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Step 11: Make predictions
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 12: Plot predictions vs real values
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Actual Google Stock Price')
plt.plot(predicted_prices, label='Predicted Stock Price')
plt.title('Google Stock Price Prediction using RNN')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
