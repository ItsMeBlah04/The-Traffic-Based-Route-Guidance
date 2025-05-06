import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('scats_data.csv')

# Extract only the V00–V95 columns
value_columns = [f'V{str(i).zfill(2)}' for i in range(96)]
values = data[value_columns].values  # shape: (num_days, 96)

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Create sequences (e.g., use past 7 days to predict the 8th)
def create_daily_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])        # shape (seq_length, 96)
        y.append(data[i + seq_length])         # shape (96,)
    return np.array(X), np.array(y)

seq_length = 7
X, y = create_daily_sequences(scaled, seq_length)

print(f"X shape: {X.shape}, y shape: {y.shape}")  # should be (samples, 7, 96), (samples, 96)

# Split train/test
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build model
model = Sequential()
model.add(Input(shape=(seq_length, 96)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(96))
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# Inverse scale predictions
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)
