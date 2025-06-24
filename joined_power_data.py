import pandas as pd

# Load data
device_df = pd.read_csv("exported_csv/Device.csv")
devicedata_df = pd.read_csv("exported_csv/DeviceData.csv")

# Merge on device_id (inner join)
joined_df = pd.merge(devicedata_df, device_df, on="device_id", how="inner")

# Select only needed columns (optional)
result = joined_df[["device_id", "name", "time_id", "power_usage"]]

# Save to CSV
result.to_csv("exported_csv/joined_power_data.csv", index=False)
print("✅ Full joined CSV created: joined_power_data.csv")