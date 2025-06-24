import pandas as pd

# Load CSV files
devicedata_df = pd.read_csv("exported_csv/DeviceData.csv")
timepoint_df = pd.read_csv("exported_csv/TimePoint.csv")
weather_df = pd.read_csv("exported_csv/Weather.csv")
device_df = pd.read_csv("exported_csv/Device.csv")
room_df = pd.read_csv("exported_csv/Room.csv")

# Join devicedata + device to get room_id
df = pd.merge(devicedata_df, device_df[['device_id', 'room_id']], on='device_id', how='inner')

# Join with timepoint to get timestamp
df = pd.merge(df, timepoint_df[['time_id', 'timestamp', 'weather_id']], on='time_id', how='inner')

# Join with weather to get summary
df = pd.merge(df, weather_df[['weather_id', 'summary']], on='weather_id', how='inner')

# Join with room to get room name
df = pd.merge(df, room_df.rename(columns={'name': 'room_name'}), on='room_id', how='inner')

# Final columns needed
final_df = df[[
    'power_usage',
    'time_id',
    'device_id',
    'room_id',
    'summary',
    'timestamp',
    'room_name'
]]

# Save to CSV
final_df.to_csv("exported_csv/devicedata_for_coactive_devices.csv", index=False)
print("✅ Merged CSV created with Room info.")