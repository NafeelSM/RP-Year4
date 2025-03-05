from flask import Flask, jsonify
import numpy as np
import datetime
import joblib
import time
import threading
app = Flask(__name__)


scaler = joblib.load('scaler.pkl')
model = joblib.load('random_forest_model.pkl')
features = ['Temperature', 'Humidity', 'Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z']


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

# Real-time
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

# Flask route to start the monitoring
@app.route('/start_monitoring', methods=['GET'])
def start_monitoring():
    threading.Thread(target=monitor_vehicle, daemon=True).start()
    return jsonify({"message": "Vehicle monitoring started!"})

@app.route('/check_status', methods=['GET'])
def check_status():
    new_data = collect_data()
    new_data_values = list(new_data.values())[1:-2]
    new_data_scaled = scaler.transform([new_data_values])
    prediction = model.predict(new_data_scaled)

    if prediction == 1:
        return jsonify({"status": "Maintenance Alert", "timestamp": new_data['Timestamp']})
    else:
        return jsonify({"status": "Vehicle Operating Normally", "timestamp": new_data['Timestamp']})

if __name__ == '__main__':
    app.run(debug=True)
