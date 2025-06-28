import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple
import warnings
import csv
warnings.filterwarnings('ignore')

class SmartHomeDBNormalizer:
    def __init__(self, db_path='smarthome.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.device_mapping = self._create_device_mapping()
        self.room_mapping = self._create_room_mapping()
        
    def _create_device_mapping(self) -> Dict[str, Dict]:
        """Create logical device mapping with categories and types"""
        return {
            'Dishwasher [kW]': {
                'name': 'Dishwasher',
                'type': 'appliance',
                'category': 'energy_consumer',
                'room': 'Kitchen'
            },
            'Furnace 1 [kW]': {
                'name': 'Primary Furnace',
                'type': 'hvac',
                'category': 'energy_consumer',
                'room': 'Basement'
            },
            'Furnace 2 [kW]': {
                'name': 'Secondary Furnace',
                'type': 'hvac',
                'category': 'energy_consumer',
                'room': 'Basement'
            },
            'Home office [kW]': {
                'name': 'Home Office Electronics',
                'type': 'electronics',
                'category': 'energy_consumer',
                'room': 'Home Office'
            },
            'Fridge [kW]': {
                'name': 'Refrigerator',
                'type': 'appliance',
                'category': 'energy_consumer',
                'room': 'Kitchen'
            },
            'Wine cellar [kW]': {
                'name': 'Wine Cellar Cooling',
                'type': 'appliance',
                'category': 'energy_consumer',
                'room': 'Wine Cellar'
            },
            'Garage door [kW]': {
                'name': 'Garage Door Motor',
                'type': 'motor',
                'category': 'energy_consumer',
                'room': 'Garage'
            },
            'Kitchen 12 [kW]': {
                'name': 'Kitchen Outlet Circuit 1',
                'type': 'circuit',
                'category': 'energy_consumer',
                'room': 'Kitchen'
            },
            'Kitchen 14 [kW]': {
                'name': 'Kitchen Outlet Circuit 2',
                'type': 'circuit',
                'category': 'energy_consumer',
                'room': 'Kitchen'
            },
            'Kitchen 38 [kW]': {
                'name': 'Kitchen Major Appliances',
                'type': 'circuit',
                'category': 'energy_consumer',
                'room': 'Kitchen'
            },
            'Barn [kW]': {
                'name': 'Barn Electronics',
                'type': 'electronics',
                'category': 'energy_consumer',
                'room': 'Barn'
            },
            'Well [kW]': {
                'name': 'Water Well Pump',
                'type': 'pump',
                'category': 'energy_consumer',
                'room': 'Utility'
            },
            'Microwave [kW]': {
                'name': 'Microwave Oven',
                'type': 'appliance',
                'category': 'energy_consumer',
                'room': 'Kitchen'
            },
            'Living room [kW]': {
                'name': 'Living Room Electronics',
                'type': 'electronics',
                'category': 'energy_consumer',
                'room': 'Living Room'
            },
            'Solar [kW]': {
                'name': 'Solar Panel System',
                'type': 'solar',
                'category': 'energy_generator',
                'room': 'Roof'
            }
        }
    
    def _create_room_mapping(self) -> Dict[str, Dict]:
        """Create room mapping with floor information"""
        return {
            'Kitchen': {'floor': 1},
            'Living Room': {'floor': 1},
            'Home Office': {'floor': 1},
            'Garage': {'floor': 1},
            'Basement': {'floor': 0},
            'Wine Cellar': {'floor': 0},
            'Utility': {'floor': 0},
            'Barn': {'floor': 1},
            'Roof': {'floor': 2}
        }
    
    def create_normalized_tables(self):
        """Create all normalized tables"""
        cursor = self.conn.cursor()
        
        # Drop existing tables
        tables = ['DeviceData', 'TimePoint', 'Weather', 'Device', 'Room']
        for table in tables:
            cursor.execute(f'DROP TABLE IF EXISTS {table}')
        
        # Create Room table
        cursor.execute('''
            CREATE TABLE Room (
                room_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                floor INTEGER NOT NULL
            )
        ''')
        
        # Create Device table
        cursor.execute('''
            CREATE TABLE Device (
                device_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                category TEXT NOT NULL,
                room_id INTEGER,
                FOREIGN KEY (room_id) REFERENCES Room (room_id)
            )
        ''')
        
        # Create Weather table
        cursor.execute('''
            CREATE TABLE Weather (
                weather_id INTEGER PRIMARY KEY AUTOINCREMENT,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                summary TEXT,
                wind REAL,
                cloud REAL,
                precip_intensity REAL,
                precip_probability REAL,
                apparent_temperature REAL,
                dew_point REAL,
                wind_bearing INTEGER,
                visibility REAL
            )
        ''')
        
        # Create TimePoint table
        cursor.execute('''
            CREATE TABLE TimePoint (
                time_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                house_overall_usage REAL,
                solar_generation REAL,
                weather_id INTEGER,
                FOREIGN KEY (weather_id) REFERENCES Weather (weather_id)
            )
        ''')
        
        # Create DeviceData table
        cursor.execute('''
            CREATE TABLE DeviceData (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id INTEGER,
                power_usage REAL,
                time_id INTEGER,
                FOREIGN KEY (device_id) REFERENCES Device (device_id),
                FOREIGN KEY (time_id) REFERENCES TimePoint (time_id)
            )
        ''')
        
        self.conn.commit()
        print("✅ All normalized tables created successfully!")
    
    def populate_rooms(self):
        """Populate Room table"""
        cursor = self.conn.cursor()
        
        for room_name, room_info in self.room_mapping.items():
            cursor.execute(
                'INSERT INTO Room (name, floor) VALUES (?, ?)',
                (room_name, room_info['floor'])
            )
        
        self.conn.commit()
        print("✅ Room table populated successfully!")
    
    def populate_devices(self):
        """Populate Device table"""
        cursor = self.conn.cursor()
        
        # Get room_id mapping
        cursor.execute('SELECT room_id, name FROM Room')
        room_id_map = {name: room_id for room_id, name in cursor.fetchall()}
        
        for original_name, device_info in self.device_mapping.items():
            room_id = room_id_map.get(device_info['room'])
            cursor.execute(
                'INSERT INTO Device (name, type, category, room_id) VALUES (?, ?, ?, ?)',
                (device_info['name'], device_info['type'], device_info['category'], room_id)
            )
        
        self.conn.commit()
        print("✅ Device table populated successfully!")
    
    def load_and_process_data(self, file_path: str):
        """Load and process the denormalized data"""
        try:
            # Read the data - assuming space-separated format based on your sample
            df = pd.read_csv(file_path, sep='\s+', header=0)
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Create sample data for demonstration
            return self._create_sample_data()

    def _create_sample_data(self):
        """Create sample data with realistic device behavior"""
        print("📝 Creating sample data with realistic device behavior...")

        # Time setup
        # Generate timestamps from 2010 to May 2025, hourly intervals
        start_time = datetime(2010, 1, 1)
        end_time = datetime(2025, 5, 31, 23, 59, 59)  # End of May 2025
        timestamps = pd.date_range(start_time, end_time, freq='H')
        n_records = len(timestamps)

        print(f"📅 Time range: {start_time} to {end_time}")
        print(f"📊 Total records: {n_records:,} hours")
        print(f"📊 Total days: {(end_time - start_time).days + 1} days")
        print(f"📊 Total years: {end_time.year - start_time.year + 1} years")

        np.random.seed(42)  # For reproducibility

        # Weather simulation with Italian climate characteristics
        temperature = []
        summaries = []

        for ts in timestamps:
            hour = ts.hour
            month = ts.month

            # General European-style temperature by season
            if month in [12, 1, 2]:  # Winter
                temp = np.random.uniform(-10, 10)  # Balanced between negative and positive
            elif month in [6, 7, 8]:  # Summer
                temp = np.random.uniform(15, 37)
            elif month in [3, 4, 5]:  # Spring
                temp = np.random.uniform(5, 22)
            else:  # Fall (9, 10, 11)
                temp = np.random.uniform(2, 20)
            temperature.append(temp)

            # Determine weather summary based on temperature
            if temp <= 0:
                summary = 'Snowy'
            elif month in [12, 1, 2] and temp < 5:
                summary = np.random.choice(
                    ['Snowy', 'Cloudy', 'Rainy', 'Clear'],
                    p=[0.4, 0.3, 0.2, 0.1]
                )
            elif temp > 25:
                summary = 'Clear'
            elif temp > 15:
                summary = np.random.choice(['Clear', 'Cloudy', 'Rainy'], p=[0.5, 0.3, 0.2])
            elif temp > 5:
                summary = np.random.choice(['Cloudy', 'Rainy'], p=[0.6, 0.4])
            else:
                summary = np.random.choice(['Rainy', 'Cloudy'], p=[0.5, 0.5])
            summaries.append(summary)

        # DEBUG: Print stats
        print("--------------------------------------------")
        print(f"🌡️ Total winter records: {sum(1 for i, m in enumerate(timestamps) if m.month in [12, 1, 2])}")
        print(f"🌡️ Winter temps < 0°C: {sum(1 for t in temperature if t < 0)}")
        print(f"❄️ Snowy days: {summaries.count('Snowy')}")
        print(f"🌡️ Min temperature: {min(temperature):.1f}°C")
        print(f"☀️ Max temperature: {max(temperature):.1f}°C")
        print("--------------------------------------------")

        # Generate solar generation based on weather and time
        solar_generation = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            summary = summaries[i]
            
            if 6 <= hour <= 18:  # Daylight hours
                if summary == 'Clear':
                    solar = np.random.uniform(2, 8)
                elif summary == 'Cloudy':
                    solar = np.random.uniform(0.5, 3)
                else:  # Rainy/Snowy
                    solar = np.random.uniform(0, 1)
            else:
                solar = 0
            solar_generation.append(solar)

        # Additional weather parameters with Italian characteristics
        humidity = []
        precip_intensity = []
        precip_probability = []
        wind_speed = []
        cloud_cover = []
        apparent_temperature = []
        dew_point = []
        visibility = []

        for i in range(n_records):
            summary = summaries[i]
            temp = temperature[i]
            month = timestamps[i].month

            # Seasonal and weather-dependent humidity generator
            if month in [12, 1, 2]:  # Winter
                if summary in ['Snowy', 'Rainy']:
                    base_humidity = np.random.uniform(0.75, 0.85)  # Higher minimum for precipitation
                else:
                    base_humidity = np.random.uniform(0.4, 0.7)  # Moderate otherwise

            elif month in [6, 7, 8]:  # Summer
                if summary == 'Clear':
                    base_humidity = np.random.uniform(0.3, 0.6)   # Dry sunny days
                elif summary == 'Cloudy':
                    base_humidity = np.random.uniform(0.5, 0.7)   # Medium
                else:  # Rainy
                    base_humidity = np.random.uniform(0.75, 0.85)  # Higher minimum for summer storms

            elif month in [3, 4, 5]:  # Spring
                if summary == 'Clear':
                    base_humidity = np.random.uniform(0.4, 0.6)
                elif summary == 'Cloudy':
                    base_humidity = np.random.uniform(0.5, 0.75)
                else:  # Rainy
                    base_humidity = np.random.uniform(0.75, 0.85)  # Higher minimum for spring rain

            else:  # Fall
                if summary == 'Clear':
                    base_humidity = np.random.uniform(0.35, 0.6)
                elif summary == 'Cloudy':
                    base_humidity = np.random.uniform(0.5, 0.7)
                else:  # Rainy
                    base_humidity = np.random.uniform(0.75, 0.85)  # Higher minimum for fall rain

            # Option 1: Add smaller variation with bounds checking
            variation = np.random.normal(0, 0.02)  # Smaller variation
            h = np.clip(base_humidity + variation, 0.0, 1.0)  # Ensure bounds [0, 1]

            # print(h) # Correct

            # Precipitation based on weather
            if summary in ['Snowy', 'Rainy']:
                p_int = np.random.uniform(1, 8)
                p_prob = np.random.uniform(0.6, 0.9)
            elif summary == 'Cloudy':
                p_int = np.random.uniform(0, 2)
                p_prob = np.random.uniform(0.1, 0.4)
            else:  # Clear
                p_int = 0
                p_prob = np.random.uniform(0, 0.1)

            # Italian wind patterns (generally light to moderate)
            if summary == 'Rainy':
                wind = np.random.uniform(8, 20)  # Windier during storms
            elif summary == 'Clear':
                wind = np.random.uniform(2, 8)   # Light winds on clear days
            else:  # Cloudy
                wind = np.random.uniform(4, 12)  # Moderate winds

            # Cloud cover based on weather summary
            if summary == 'Clear':
                clouds = np.random.uniform(0.05, 0.25)
            elif summary == 'Cloudy':
                clouds = np.random.uniform(0.6, 0.9)
            else:  # Rainy/Snowy
                clouds = np.random.uniform(0.8, 0.98)

            humidity.append(h)
            precip_intensity.append(p_int)
            precip_probability.append(p_prob)
            wind_speed.append(wind)
            cloud_cover.append(clouds)

            summary = summaries[i]
            temp = temperature[i]
            h = humidity[i]
            w = wind_speed[i]

            # Generate logically consistent weather-dependent values
            if temp <= 0 and w > 5:
                apparent_temp = 13.12 + 0.6215 * temp - 11.37 * (w ** 0.16) + 0.3965 * temp * (w ** 0.16)
            else:
                apparent_temp = temp

            dew = temp - np.random.uniform(0, 2) if summary in ['Snowy', 'Rainy'] else temp - np.random.uniform(2, 5)
            dew = min(dew, temp)

            if summary in ['Snowy', 'Rainy']:
                vis = np.random.uniform(1, 5)
            elif summary == 'Cloudy':
                vis = np.random.uniform(5, 10)
            else:
                vis = np.random.uniform(10, 15)

            apparent_temperature.append(apparent_temp)
            dew_point.append(dew)
            visibility.append(vis)    

        # Device usage
        data = {
            'time': [int(ts.timestamp()) for ts in timestamps],
            'temperature': temperature,
            'humidity': humidity,
            'pressure': np.random.uniform(980, 1030, n_records),
            'summary': summaries,
            'wind': wind_speed,
            'cloud': cloud_cover,
            'precipIntensity': precip_intensity,
            'precipProbability': precip_probability,
            'apparentTemperature': apparent_temperature,
            'dewPoint': dew_point,
            'visibility': visibility,
            'windBearing': np.random.randint(0, 360, n_records),
            'Dishwasher [kW]': self._generate_device_usage(
                base_power=(1.2, 2.0), off_prob=0.8, n=n_records, hourly_pattern=[(18, 22, 0.3)], timestamps=timestamps),
            'Furnace 1 [kW]': self._generate_weather_dependent_device(
                base_power=(1.5, 3.0), low_temp_prob=0.8, mid_temp_prob=0.4, high_temp_prob=0.1, temps=temperature),
            'Furnace 2 [kW]': self._generate_weather_dependent_device(
                base_power=(1.0, 2.5), low_temp_prob=0.8, mid_temp_prob=0.4, high_temp_prob=0.1, temps=temperature),
            'Home office [kW]': self._generate_intermittent_usage(
                base_power=(0.1, 0.3), off_night_prob=0.8, n=n_records, ts_list=timestamps),
            'Fridge [kW]': np.random.uniform(0.1, 0.2, n_records),
            'Wine cellar [kW]': self._generate_warm_weather_usage(
                base_power=(0.3, 0.5), threshold=20, temps=temperature),
            'Garage door [kW]': np.random.choice([0, 0.5], size=n_records, p=[0.99, 0.01]),
            'Kitchen 12 [kW]': np.random.uniform(0.2, 0.6, n_records),
            'Kitchen 14 [kW]': np.random.uniform(0.3, 1.0, n_records),
            'Kitchen 38 [kW]': self._generate_device_usage(
                base_power=(2.0, 5.0), off_prob=0.8, n=n_records, hourly_pattern=[(17, 21, 0.3)], timestamps=timestamps),
            'Barn [kW]': np.random.uniform(0.1, 0.5, n_records),
            'Well [kW]': np.random.choice([0, 1.2], size=n_records, p=[0.8, 0.2]),
            'Microwave [kW]': self._generate_device_usage(
                base_power=(0.8, 1.5), off_prob=0.9, n=n_records, hourly_pattern=[(18, 22, 0.2)], timestamps=timestamps),
            'Living room [kW]': self._generate_intermittent_usage(
                base_power=(0.1, 0.4), off_night_prob=0.8, n=n_records, ts_list=timestamps),
            'Fan [kW]': self._generate_intermittent_usage(
                base_power=(0.02, 0.1), off_night_prob=0.7, n=n_records, ts_list=timestamps),

            'Solar [kW]': [-g if g > 0 else 0 for g in solar_generation],
        }

        df = pd.DataFrame(data)

        # Inject defects
        # self._inject_overcapacity(df, 'Kitchen 38 [kW]', max_power=5.0)
        # self._inject_wrong_season_usage(df, 'Furnace 1 [kW]', df['temperature'], threshold=20)
        # self._inject_solar_failure(df, 'Solar [kW]', df['summary'])
        # self._inject_sensor_drift(df, 'humidity', drift_rate=0.005)
        # self._inject_intermittent_outage(df, 'Fan [kW]', outage_prob=0.02)

        # Drop missing timestamps and reset index
        df = self._inject_missing_timestamps(timestamps, df, missing_prob=0.002)

        # Final house power usage calculation
        device_columns = [
            'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
            'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
            'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
            'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 'Microwave [kW]',
            'Living room [kW]', 'Fan [kW]'
        ]
        df['House overall [kW]'] = df[device_columns].sum(axis=1)

        # Safety check
        lengths = [len(df[col]) for col in df.columns]
        assert len(set(lengths)) == 1, f"Inconsistent data lengths: {set(lengths)}"

        print(f"✅ Sample data created with defects! Shape: {df.shape}")
        return df

    # Helper methods
    def _generate_device_usage(self, base_power, off_prob, n, hourly_pattern=None, timestamps=None):
        usage = []
        for _ in range(n):
            if np.random.rand() < off_prob:
                usage.append(0)
            else:
                usage.append(np.random.uniform(*base_power))
        if hourly_pattern and timestamps is not None:
            for start_hour, end_hour, on_prob in hourly_pattern:
                for i, ts in enumerate(timestamps):
                    if start_hour <= ts.hour <= end_hour:
                        usage[i] = np.random.choice([0, np.random.uniform(*base_power)], p=[1-on_prob, on_prob])
        return usage

    def _generate_weather_dependent_device(self, base_power, low_temp_prob, mid_temp_prob, high_temp_prob, temps):
        """Adjust usage probability based on temperature."""
        usage = []
        for t in temps:
            if t < 5:
                prob = low_temp_prob
            elif t < 15:
                prob = mid_temp_prob
            else:
                prob = high_temp_prob
            usage.append(np.random.uniform(*base_power) if np.random.rand() < prob else 0)
        return usage

    def _generate_intermittent_usage(self, base_power, off_night_prob, n, ts_list):
        """Add night-time off periods."""
        usage = []
        for i in range(n):
            hour = ts_list[i].hour
            if 0 <= hour < 6:
                val = 0 if np.random.rand() < off_night_prob else np.random.uniform(*base_power)
            else:
                val = np.random.uniform(*base_power)
            usage.append(val)
        return usage

    def _generate_warm_weather_usage(self, base_power, threshold, temps):
        """Increase usage when it's warm."""
        usage = []
        for t in temps:
            if t > threshold:
                val = np.random.uniform(*base_power) if np.random.rand() < 0.5 else 0
            else:
                val = np.random.uniform(*base_power[:1])  # Lower usage
            usage.append(val)
        return usage

    # --- Defect Injection Helper Methods ---
    def _inject_overcapacity(self, df, column, max_power):
        """Simulate a device exceeding its normal power limit"""
        idxs = df.sample(frac=0.05).index  # 5% of records
        df.loc[idxs, column] *= np.random.uniform(1.5, 2.0)  # Boost by 50–100%
        df[column] = np.clip(df[column], None, max_power * 2)  # Cap it realistically
        print(f"🔥 Injected overcapacity events in '{column}'")

    def _inject_wrong_season_usage(self, df, column, temperature, threshold=20):
        """Simulate furnace use in warm weather"""
        warm_days = (temperature > threshold)
        idxs = df[warm_days].sample(frac=0.1).index  # 10% of warm days
        df.loc[idxs, column] = np.random.uniform(1.0, 3.0)
        print(f"❄️ Injected wrong-season usage in '{column}'")

    def _inject_solar_failure(self, df, column, summary):
        """Simulate solar panel failure in sunny weather"""
        sunny = (summary == 'Clear')
        idxs = df[sunny].sample(frac=0.1).index  # 10% of sunny records
        df.loc[idxs, column] = 0  # Zero generation despite sun
        print(f"☀️ Injected solar failures in sunny weather")

    def _inject_sensor_drift(self, df, column, drift_rate=0.005):
        """Simulate gradual sensor drift upward"""
        drift = np.linspace(0, drift_rate * len(df), len(df))
        df[column] += drift
        df[column] = np.clip(df[column], 0, 0.98)
        print(f"💧 Injected sensor drift in '{column}'")

    def _inject_intermittent_outage(self, df, column, outage_prob=0.02):
        """Simulate random outages where device stops reporting"""
        outage_mask = np.random.rand(len(df)) < outage_prob
        df.loc[outage_mask, column] = 0
        print(f"⚡ Injected intermittent outages in '{column}'")

    def _inject_missing_timestamps(self, timestamps, df, missing_prob=0.003):
        """Simulate missing rows due to logging errors"""
        missing_mask = np.random.rand(len(df)) < missing_prob
        df.drop(df[missing_mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"🕒 Injected missing timestamps")
        return df
    
    # Error
    def populate_weather_and_timepoint(self, df):
        """Populate Weather and TimePoint tables in sync"""
        cursor = self.conn.cursor()
        
        timepoint_data = []
        
        for _, row in df.iterrows():
            # Step 1: Insert Weather record
            weather_record = (
                row.get('temperature', 20),
                row.get('humidity', 0.5),
                row.get('pressure', 1013),
                row.get('summary', 'Clear'),
                row.get('wind', 5),
                row.get('cloud', 0.5),
                row.get('precipIntensity', 0),
                row.get('precipProbability', 0),
                row.get('apparentTemperature', 20),
                row.get('dewPoint', 10),
                row.get('windBearing', 180),
                row.get('visibility', 10)
            )
            
            cursor.execute('''
                INSERT INTO Weather (
                    temperature, humidity, pressure, summary,
                    wind, cloud, precip_intensity,
                    precip_probability, apparent_temperature,
                    dew_point, wind_bearing, visibility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', weather_record)
            weather_id = cursor.lastrowid  # Get ID of inserted Weather record

            # Step 2: Insert corresponding TimePoint with correct weather_id
            timepoint_record = (
                int(row.get('time', datetime.now().timestamp())),
                row.get('House overall [kW]', 0),
                row.get('Solar [kW]', 0),
                weather_id  # This ensures alignment
            )
            timepoint_data.append(timepoint_record)

        # Insert all timepoints at once
        cursor.executemany('''
            INSERT INTO TimePoint (
                timestamp, house_overall_usage, solar_generation, weather_id
            ) VALUES (?, ?, ?, ?)
        ''', timepoint_data)

        self.conn.commit()
        print("✅ Weather and TimePoint tables populated successfully!")
    
    def populate_device_data(self, df):
        """Populate DeviceData table"""
        cursor = self.conn.cursor()
        
        # Get device ID mapping
        cursor.execute('SELECT device_id, name FROM Device')
        device_rows = cursor.fetchall()
        
        # Create reverse mapping from original names to device IDs
        device_id_map = {}
        for device_id, device_name in device_rows:
            for orig_name, device_info in self.device_mapping.items():
                if device_info['name'] == device_name:
                    device_id_map[orig_name] = device_id
                    break
        
        # Get time IDs
        cursor.execute('SELECT time_id FROM TimePoint ORDER BY time_id')
        time_ids = [row[0] for row in cursor.fetchall()]
        
        device_data = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            time_id = time_ids[i]
            
            for orig_col_name, device_id in device_id_map.items():
                if orig_col_name in df.columns:
                    power_usage = row.get(orig_col_name, 0)
                    if pd.notna(power_usage):
                        device_data.append((device_id, float(power_usage), time_id))
        
        cursor.executemany('''
            INSERT INTO DeviceData (device_id, power_usage, time_id)
            VALUES (?, ?, ?)
        ''', device_data)
        
        self.conn.commit()
        print("✅ DeviceData table populated successfully!")
    
    def normalize_database(self, file_path: str = None):
        """Complete normalization process"""
        print("🚀 Starting database normalization process...")
        
        # Step 1: Create tables
        self.create_normalized_tables()
        
        # Step 2: Populate static tables
        self.populate_rooms()
        self.populate_devices()
        
        # Step 3: Load and process data
        if file_path:
            df = self.load_and_process_data(file_path)
        else:
            df = self._create_sample_data()

        print(df) # Correct
    
        # Step 4: Populate dynamic tables
        self.populate_weather_and_timepoint(df)
        self.populate_device_data(df)
        
        print("🎉 Database normalization completed successfully!")
        self.print_table_stats()
    
    def print_table_stats(self):
        """Print statistics about each table"""
        cursor = self.conn.cursor()
        
        tables = ['Room', 'Device', 'Weather', 'TimePoint', 'DeviceData']
        
        print("\n📊 Table Statistics:")
        print("-" * 40)
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f"{table:12}: {count:6,} records")
    
    def export_tables_to_csv(self, output_dir='exported_csv'):
        """Export all tables to CSV files"""
        import os
        cursor = self.conn.cursor()

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            # Get column names
            column_names = [description[0] for description in cursor.description]

            # Write to CSV
            file_path = os.path.join(output_dir, f"{table_name}.csv")
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(column_names)  # Write header
                writer.writerows(rows)        # Write data
            print(f"✅ Exported table '{table_name}' to {file_path}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# Usage example and demonstration
def main():
    """Main function to demonstrate the normalization process"""
    print("🏠 Smart Home Database Normalization System")
    print("=" * 50)
    
    # Initialize the normalizer
    normalizer = SmartHomeDBNormalizer('exported_csv/smarthome_normalized.db')
    
    try:
        # Normalize the database (using sample data)
        normalizer.normalize_database()

        # Export to CSV
        normalizer.export_tables_to_csv()
        
        print("\n✅ Process completed successfully!")
        print(f"📁 Database saved as: {normalizer.db_path}")
        print("\n💡 You can now use this normalized database for your queries!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        normalizer.close()

if __name__ == "__main__":
    main()