import csv
from pathlib import Path

# Namespace prefix
NS = "http://example.org/"

def room_to_ttl(row):
    """Convert a row from rooms.csv to Turtle."""
    return f"""
  ex:R{row['room_id']} a ex:Room ;
  ex:roomId {row['room_id']} ;
  ex:name "{row['name']}" ;
  {"ex:floor " + row['floor'] if 'floor' in row else ""} .
"""

def device_to_ttl(row):
    """Convert a row from devices.csv to Turtle."""
    return f"""
  ex:D{row['device_id']} a ex:Device ;
  ex:deviceId {row['device_id']} ;
  ex:name "{row['name']}" ;
  ex:type "{row['type']}" ;
  ex:category "{row['category']}" ;
  ex:locatedIn ex:R{row['room_id']} .
"""

def timepoint_to_ttl(row):
    """Convert a row from timepoints.csv to Turtle."""
    return f"""
  ex:TP{row['time_id']} a ex:TimePoint ;
  ex:timeId {row['time_id']} ;
  ex:timestamp {row['timestamp']} ;
  ex:solarGeneration {row['solar_generation']} ;
  ex:houseOverallUsage {row['house_overall_usage']} ;
  ex:hasWeather ex:W{row['weather_id']} .
"""

def devicedata_to_ttl(row):
    """Convert a row from device_data.csv to Turtle."""
    return f"""
  ex:DD{row['data_id']} a ex:DeviceData ;
  ex:dataId {row['data_id']} ;
  ex:powerUsage {row['power_usage']} ;
  ex:isProducedBy ex:D{row['device_id']} ;
  ex:recordsAt ex:TP{row['time_id']} .
"""

def weather_to_ttl(row):
    """Convert a row from weather.csv to Turtle."""
    return f"""
  ex:W{row['weather_id']} a ex:Weather ;
  ex:weatherId {row['weather_id']} ;
  ex:summary "{row['summary']}" ;
  ex:temperature {row['temperature']} ;
  ex:humidity {row['humidity']} ;
  ex:windSpeed {row['wind_speed']} .
"""

def main():
    # Output directory for intermediate TTL files
    output_dir = Path("exported_ttl")
    output_dir.mkdir(exist_ok=True)

    # Process Room
    with open("exported_csv/Room.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_dir / "Room.ttl", "w") as ttlfile:
            for row in reader:
                ttlfile.write(room_to_ttl(row))

    # Process Devices
    with open("exported_csv/Device.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_dir / "Device.ttl", "w") as ttlfile:
            for row in reader:
                ttlfile.write(device_to_ttl(row))

    # Process TimePoints
    with open("exported_csv/TimePoint.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_dir / "TimePoint.ttl", "w") as ttlfile:
            for row in reader:
                ttlfile.write(timepoint_to_ttl(row))

    # Process DeviceData
    with open("exported_csv/DeviceData.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_dir / "DeviceData.ttl", "w") as ttlfile:
            for row in reader:
                ttlfile.write(devicedata_to_ttl(row))

    # Process Weather
    with open("exported_csv/Weather.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_dir / "Weather.ttl", "w") as ttlfile:
            for row in reader:
                ttlfile.write(weather_to_ttl(row))

    # Merge all TTL files into one
    with open(output_dir / "full_dataset.ttl", "w") as merged_file:
        for filename in ["Room.ttl", "Device.ttl", "TimePoint.ttl", "DeviceData.ttl", "Weather.ttl"]:
            with open(output_dir / filename, "r") as infile:
                merged_file.write(infile.read())

if __name__ == "__main__":
    main()