import pandas as pd

# Load CSV files
devicedata_df = pd.read_csv("exported_csv/DeviceData.csv")
timepoint_df = pd.read_csv("exported_csv/TimePoint.csv")
weather_df = pd.read_csv("exported_csv/Weather.csv")

# Join devicedata + timepoint
df = pd.merge(devicedata_df, timepoint_df[['time_id', 'timestamp', 'weather_id']], on='time_id', how='inner')

# Join with weather
df = pd.merge(df, weather_df, on='weather_id', how='inner')

# Final columns
final_df = df[[
    'power_usage', 'timestamp', 'temperature',
    'time_id',  'weather_id', 'summary', 'device_id'
]]

# Save to CSV
final_df.to_csv("exported_csv/devicedata_with_weather_and_time.csv", index=False)
print("✅ Merged CSV created.")
