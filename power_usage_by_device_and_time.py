import pandas as pd

# Load CSV files
device_df = pd.read_csv("exported_csv/Device.csv")        # device_id, name, type, category, room_id
devicedata_df = pd.read_csv("exported_csv/DeviceData.csv") # data_id, device_id, power_usage, time_id
timepoint_df = pd.read_csv("exported_csv/TimePoint.csv")   # time_id, timestamp, ...
room_df = pd.read_csv("exported_csv/Room.csv")            # room_id, name (as room_name)

# Merge devicedata + device
df = pd.merge(devicedata_df, device_df, on="device_id", how="inner")

# Merge with timepoint
df = pd.merge(df, timepoint_df[['time_id', 'timestamp']], on="time_id", how="inner")

# Merge with room
df = pd.merge(df, room_df.rename(columns={'name': 'room_name'}), on="room_id", how="inner")

# Filter only Dishwasher
df_dishwasher = df[df['name'] == 'Dishwasher']

# Select final columns
final_df = df_dishwasher[[
    'device_id', 'name', 'time_id', 'power_usage',
    'timestamp', 'room_id', 'room_name'
]]

# Save to CSV
final_df.to_csv("exported_csv/power_usage_by_device_and_time.csv", index=False)
print("✅ Denormalized CSV created: power_usage_by_device_and_time.csv")
