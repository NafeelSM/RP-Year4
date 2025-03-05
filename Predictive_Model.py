import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv('vehicle_data_with.csv')

#  datetime type
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['maintenance_needed'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])

# Feature columns
features = ['Temperature', 'Humidity', 'Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
X = data[features]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df['Timestamp'] = data['Timestamp']

# The target column
y = data['maintenance_needed']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(scaled_df[features], y, test_size=0.2, random_state=42)

# Train modal
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate and Save
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
model_filename = 'random_forest_model.pkl'
scaler_filename = 'scaler.pkl'
joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)
print(f"Model Saved!")
