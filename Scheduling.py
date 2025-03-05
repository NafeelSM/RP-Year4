import pandas as pd
import numpy as np
import time
import datetime
import joblib

# Load Dataset
data = pd.read_csv('vehicle_data_with.csv')

# Datetime type
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['maintenance_needed'] = np.random.choice([0, 1], size=len(data),
                                              p=[0.95, 0.05])

# Feature columns
features = ['Temperature', 'Humidity', 'Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
X = data[features]

# Standardizing the features
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)
scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df['Timestamp'] = data['Timestamp']

# The target column
y = data['maintenance_needed']

# Load model
model = joblib.load('random_forest_model.pkl')

######## get from sensor ++LIVE++
def collect_data():
    current_time = datetime.datetime.now().strftime('%m/%d/%Y %H:%M')
    return {
        'Timestamp': current_time,
        'Temperature': np.random.uniform(25, 35),
        'Humidity': np.random.uniform(50, 80),
        'Accel X': np.random.uniform(0.4, 0.6),
        'Accel Y': np.random.uniform(-1, -0.8),
        'Accel Z': np.random.uniform(8, 9),
        'Gyro X': np.random.uniform(-0.1, 0),
        'Gyro Y': np.random.uniform(-0.12, -0.1),
        'Gyro Z': np.random.uniform(0, 0.05),
        'Latitude': np.random.uniform(6.46, 6.48),
        'Longitude': np.random.uniform(80.10, 80.12)
    }

# Real-time monitoring function
def monitor_vehicle():
    while True:

        new_data = collect_data()
        new_data_values = list(new_data.values())[1:-2]
        new_data_scaled = scaler.transform([new_data_values])
        prediction = model.predict(new_data_scaled)

        if prediction == 1:
            print(f"Maintenance Alert at {new_data['Timestamp']}: Vehicle requires maintenance!")
        else:
            print(f"Vehicle is operating normally at {new_data['Timestamp']}.")

        time.sleep(60)


monitor_vehicle()
