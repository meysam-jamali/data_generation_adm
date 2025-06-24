import pandas as pd

# Load CSV files
device_df = pd.read_csv("exported_csv/Device.csv")
room_df = pd.read_csv("exported_csv/Room.csv")
devicedata_df = pd.read_csv("exported_csv/DeviceData.csv")
timepoint_df = pd.read_csv("exported_csv/TimePoint.csv")

# Merge device -> room
device_with_room = pd.merge(device_df, room_df.rename(columns={'name': 'room_name'}), on='room_id', how='inner')

# Merge devicedata -> device + room
df = pd.merge(devicedata_df, device_with_room, on='device_id', how='inner')

# Merge with timepoint to get solar_generation and timestamp
df = pd.merge(df, timepoint_df[['time_id', 'timestamp', 'solar_generation']], on='time_id', how='inner')

# Select final columns
final_df = df[[
     
    'device_id', 'room_id', 'power_usage', 'solar_generation', 'room_name',  'timestamp'
]]

# Save to CSV
final_df.to_csv("exported_csv/room_power_with_solar.csv", index=False)
print("✅ Merged CSV created.")