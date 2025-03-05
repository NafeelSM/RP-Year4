import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

#   data CSV
data = pd.read_csv('vehicle_data_with.csv')

#  timestamp to datetime type
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(data['Timestamp'], data['Temperature'], label='Temperature (°C)')
plt.plot(data['Timestamp'], data['Humidity'], label='Humidity (%)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Time Series of Vehicle Sensor Data')
plt.legend()
plt.show()

# Apply Time Series Decomposition (Trend, Seasonality, Residuals) using statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['Temperature'], model='additive', period=60)  # 60 to indicate daily seasonality
result.plot()



scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Temperature', 'Humidity', 'Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z']])

scaled_df = pd.DataFrame(scaled_data, columns=['Temperature', 'Humidity', 'Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z'])
scaled_df['Timestamp'] = data['Timestamp']

# Check the scaled data
print(scaled_df.head())
