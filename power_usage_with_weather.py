import pandas as pd

# Load CSV files
timepoint_df = pd.read_csv("exported_csv/TimePoint.csv")
weather_df = pd.read_csv("exported_csv/Weather.csv")
devicedata_df = pd.read_csv("exported_csv/DeviceData.csv")

# Merge timepoint + weather
df = pd.merge(timepoint_df, weather_df, on="weather_id", how="inner")

# Merge with devicedata
df = pd.merge(df, devicedata_df, on="time_id", how="inner")

# Final columns
final_df = df[[
    "timestamp", "temperature", "summary", "time_id", "weather_id",
    "device_id", "power_usage"
]]

# Save to CSV
final_df.to_csv("exported_csv/power_usage_with_weather.csv", index=False)
print("✅ Merged CSV created.")