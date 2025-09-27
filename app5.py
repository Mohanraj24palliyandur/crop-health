import pandas as pd

# Simulate the sensor data
sensor_data = {
    'timestamp': ['2025-09-01', '2025-09-02', '2025-09-03'],
    'soil_moisture': [0.35, 0.42, 0.50],  # Value between 0 and 1
    'air_temperature': [28.5, 29.0, 30.2],  # in Celsius
    'humidity': [60, 62, 64],  # in %
}

df = pd.DataFrame(sensor_data)
print(df)
