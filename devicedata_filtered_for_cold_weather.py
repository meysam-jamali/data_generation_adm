import pandas as pd

# Load CSV files
devicedata_df = pd.read_csv("exported_csv/DeviceData.csv")
timepoint_df = pd.read_csv("exported_csv/TimePoint.csv")
weather_df = pd.read_csv("exported_csv/Weather.csv")

# Join devicedata with timepoint using 'time_id'
df = pd.merge(devicedata_df, timepoint_df[['time_id', 'timestamp', 'weather_id']], on='time_id', how='inner')

# Join with weather using 'weather_id'
df = pd.merge(df, weather_df[['weather_id', 'temperature', 'humidity', 'wind_speed', 'summary']], on='weather_id', how='inner')

# Final columns needed for this query
final_df = df[[
    'power_usage',
    'wind_speed',
    'temperature',
    'humidity',
    'device_id',
    'summary'
]]

# Save to CSV
final_df.to_csv("exported_csv/devicedata_filtered_for_cold_weather.csv", index=False)
print("✅ Merged CSV created.")
