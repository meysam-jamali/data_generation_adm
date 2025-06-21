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
                wind_speed REAL,
                cloud_cover REAL,
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
    
    # def _create_sample_data(self):
    #     """Create sample data with realistic device behavior"""
    #     print("📝 Creating sample data with realistic device behavior...")
        
    #     # Time setup
    #     end_time = datetime.now()
    #     start_time = end_time - timedelta(days=30)
    #     timestamps = pd.date_range(start_time, end_time, freq='H')
    #     n_records = len(timestamps)

    #     np.random.seed(42)

    #     # Weather simulation with logical constraints
    #     temperature = np.random.uniform(-10, 40, n_records)
    #     summary_options = ['Clear', 'Cloudy', 'Rainy', 'Snowy']
    #     summaries = []
    #     solar_generation = []

    #     for i in range(n_records):
    #         if temperature[i] > 20 and np.random.rand() > 0.3:
    #             summaries.append('Clear')
    #             solar_generation.append(np.random.uniform(2, 5))
    #         elif temperature[i] > 10:
    #             summaries.append(np.random.choice(['Cloudy', 'Clear']))
    #             solar_generation.append(np.random.uniform(0.5, 3))
    #         elif temperature[i] <= 0:
    #             summaries.append(np.random.choice(['Snowy', 'Cloudy']))
    #             solar_generation.append(np.random.uniform(0, 1))
    #         else:
    #             summaries.append(np.random.choice(['Rainy', 'Cloudy']))
    #             solar_generation.append(np.random.uniform(0, 0.5))

    #     # Realistic power usage per device
    #     data = {
    #         'time': [int(ts.timestamp()) for ts in timestamps],
    #         'House overall [kW]': np.zeros(n_records),  # Will compute later

    #         # Energy consumers
    #         'Dishwasher [kW]': np.random.choice([0, np.random.uniform(1.2, 2.0)], size=n_records, p=[0.8, 0.2]),
    #         'Furnace 1 [kW]': np.random.choice([0, np.random.uniform(1.5, 3.0)], size=n_records, p=[0.6, 0.4]),
    #         'Furnace 2 [kW]': np.random.choice([0, np.random.uniform(1.0, 2.5)], size=n_records, p=[0.6, 0.4]),
    #         'Home office [kW]': np.random.uniform(0.1, 0.3, n_records),
    #         'Fridge [kW]': np.random.uniform(0.1, 0.2, n_records),
    #         'Wine cellar [kW]': np.random.uniform(0.2, 0.4, n_records),
    #         'Garage door [kW]': np.random.choice([0, 0.5], size=n_records, p=[0.95, 0.05]),
    #         'Kitchen 12 [kW]': np.random.uniform(0.2, 0.6, n_records),
    #         'Kitchen 14 [kW]': np.random.uniform(0.3, 1.0, n_records),
    #         'Kitchen 38 [kW]': np.random.choice([0, np.random.uniform(1.5, 3.0)], size=n_records, p=[0.8, 0.2]),  # Oven?
    #         'Barn [kW]': np.random.uniform(0.1, 0.5, n_records),
    #         'Well [kW]': np.random.choice([0, 1.2], size=n_records, p=[0.8, 0.2]),
    #         'Microwave [kW]': np.random.choice([0, np.random.uniform(0.8, 1.5)], size=n_records, p=[0.9, 0.1]),
    #         'Living room [kW]': np.random.uniform(0.1, 0.4, n_records),

    #         # Solar
    #         'Solar [kW]': [-g if g > 0 else 0 for g in solar_generation],  # Negative means generation

    #         # Weather
    #         'temperature': temperature,
    #         'humidity': np.random.uniform(0.3, 0.9, n_records),
    #         'pressure': np.random.uniform(980, 1030, n_records),
    #         'summary': summaries,
    #         'windSpeed': np.random.uniform(0, 20, n_records),
    #         'cloudCover': np.random.uniform(0, 1, n_records),
    #         'precipIntensity': np.random.uniform(0, 10, n_records),
    #         'precipProbability': np.random.uniform(0, 1, n_records),
    #         'apparentTemperature': np.random.uniform(-15, 45, n_records),
    #         'dewPoint': np.random.uniform(-20, 25, n_records),
    #         'windBearing': np.random.randint(0, 360, n_records),
    #         'visibility': np.random.uniform(5, 15, n_records)
    #     }

    #     # Calculate house overall usage as sum of all device usages
    #     device_columns = [
    #         'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
    #         'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
    #         'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
    #         'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 'Microwave [kW]',
    #         'Living room [kW]'
    #     ]
    #     data['House overall [kW]'] = pd.DataFrame(data)[device_columns].sum(axis=1)

    #     df = pd.DataFrame(data)
    #     print(f"✅ Sample data created! Shape: {df.shape}")
    #     return df
    

    def _create_sample_data(self):
        """Create sample data with realistic device behavior"""
        print("📝 Creating sample data with realistic device behavior...")
        
        # Time setup
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        timestamps = pd.date_range(start_time, end_time, freq='H')
        n_records = len(timestamps)

        np.random.seed(42)  # For reproducibility

        # Weather simulation with logical constraints
        temperature = []
        summaries = []
        solar_generation = []

        for ts in timestamps:
            hour = ts.hour
            month = ts.month

            # Seasonal temperature distribution
            if month in [12, 1, 2]:  # Winter
                temp = np.random.uniform(-15, 5)
            elif month in [6, 7, 8]:  # Summer
                temp = np.random.uniform(15, 35)
            else:  # Spring/Fall
                temp = np.random.uniform(0, 20)
            temperature.append(temp)

            # Determine weather summary based on temperature and randomness
            if temp > 20 and np.random.rand() > 0.3:
                summaries.append('Clear')
                solar_generation.append(np.random.uniform(2, 5))
            elif temp > 10:
                summaries.append(np.random.choice(['Cloudy', 'Clear']))
                solar_generation.append(np.random.uniform(0.5, 3))
            elif temp <= 2:  # Allow snow up to 2°C if humid
                if np.random.rand() < 0.7 and np.random.uniform(0.3, 0.9) > 0.8:
                    summaries.append('Snowy')
                else:
                    summaries.append('Cloudy')
                solar_generation.append(np.random.uniform(0, 1))
            else:
                summaries.append(np.random.choice(['Rainy', 'Cloudy']))
                solar_generation.append(np.random.uniform(0, 0.5))

        # Additional correlated weather parameters
        humidity = []
        precip_intensity = []
        precip_probability = []
        wind_speed = []
        cloud_cover = []

        for i in range(n_records):
            weather_summary = summaries[i]
            temp = temperature[i]

            # Humidity logic
            if weather_summary in ['Snowy', 'Rainy']:
                h = np.random.uniform(0.7, 0.95)
                p_int = np.random.uniform(2, 10)
                p_prob = np.random.uniform(0.5, 1.0)
            elif weather_summary == 'Cloudy':
                h = np.random.uniform(0.5, 0.8)
                p_int = np.random.uniform(0.1, 1)
                p_prob = np.random.uniform(0.1, 0.4)
            else:
                h = np.random.uniform(0.3, 0.6)
                p_int = 0
                p_prob = 0

            humidity.append(h)
            precip_intensity.append(p_int)
            precip_probability.append(p_prob)
            wind_speed.append(np.random.uniform(0, 20))
            cloud_cover.append(np.random.uniform(0, 1))

        # Realistic power usage per device
        data = {
            'time': [int(ts.timestamp()) for ts in timestamps],
            'House overall [kW]': np.zeros(n_records),  # Will compute later

            # Energy consumers
            'Dishwasher [kW]': self._generate_device_usage(
                base_power=(1.2, 2.0), off_prob=0.8, n=n_records, hourly_pattern=[(18, 22, 0.3)]),
            'Furnace 1 [kW]': self._generate_weather_dependent_device(
                base_power=(1.5, 3.0), low_temp_prob=0.8, mid_temp_prob=0.4, high_temp_prob=0.1, temps=temperature),
            'Furnace 2 [kW]': self._generate_weather_dependent_device(
                base_power=(1.0, 2.5), low_temp_prob=0.8, mid_temp_prob=0.4, high_temp_prob=0.1, temps=temperature),
            'Home office [kW]': self._generate_intermittent_usage(base_power=(0.1, 0.3), off_night_prob=0.8, n=n_records, ts_list=timestamps),
            'Fridge [kW]': np.random.uniform(0.1, 0.2, n_records),
            'Wine cellar [kW]': self._generate_warm_weather_usage(base_power=(0.3, 0.5), threshold=20, temps=temperature),
            'Garage door [kW]': np.random.choice([0, 0.5], size=n_records, p=[0.99, 0.01]),
            'Kitchen 12 [kW]': np.random.uniform(0.2, 0.6, n_records),
            'Kitchen 14 [kW]': np.random.uniform(0.3, 1.0, n_records),
            'Kitchen 38 [kW]': self._generate_device_usage(
                base_power=(2.0, 5.0), off_prob=0.8, n=n_records, hourly_pattern=[(17, 21, 0.3)]),  # Oven
            'Barn [kW]': np.random.uniform(0.1, 0.5, n_records),
            'Well [kW]': np.random.choice([0, 1.2], size=n_records, p=[0.8, 0.2]),
            'Microwave [kW]': self._generate_device_usage(
                base_power=(0.8, 1.5), off_prob=0.9, n=n_records, hourly_pattern=[(18, 22, 0.2)]),
            'Living room [kW]': self._generate_intermittent_usage(base_power=(0.1, 0.4), off_night_prob=0.8, n=n_records, ts_list=timestamps),
            'Fan [kW]': self._generate_intermittent_usage(base_power=(0.02, 0.1), off_night_prob=0.7, n=n_records, ts_list=timestamps),

            # Solar
            'Solar [kW]': [-g if g > 0 else 0 for g in solar_generation],  # Negative means generation

            # Weather
            'temperature': temperature,
            'humidity': humidity,
            'pressure': np.random.uniform(980, 1030, n_records),
            'summary': summaries,
            'windSpeed': wind_speed,
            'cloudCover': cloud_cover,
            'precipIntensity': precip_intensity,
            'precipProbability': precip_probability,
            'apparentTemperature': np.random.uniform(-15, 45, n_records),
            'dewPoint': np.random.uniform(-20, 25, n_records),
            'windBearing': np.random.randint(0, 360, n_records),
            'visibility': np.random.uniform(5, 15, n_records)
        }

        # Calculate house overall usage as sum of all device usages
        device_columns = [
            'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
            'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
            'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
            'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 'Microwave [kW]',
            'Living room [kW]', 'Fan [kW]'
        ]
        data['House overall [kW]'] = pd.DataFrame(data)[device_columns].sum(axis=1)

        df = pd.DataFrame(data)

        # Now inject defects
        self._inject_overcapacity(df, 'Kitchen 38 [kW]', max_power=5.0)
        self._inject_wrong_season_usage(df, 'Furnace 1 [kW]', df['temperature'], threshold=20)
        self._inject_solar_failure(df, 'Solar [kW]', df['summary'])
        self._inject_sensor_drift(df, 'humidity', drift_rate=0.005)
        self._inject_intermittent_outage(df, 'Fan [kW]', outage_prob=0.02)
        self._inject_missing_timestamps(timestamps, df)

        # Recalculate house overall usage after defects
        device_columns = [
            'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
            'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
            'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
            'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 'Microwave [kW]',
            'Living room [kW]', 'Fan [kW]'
        ]
        df['House overall [kW]'] = df[device_columns].sum(axis=1)

        print(f"✅ Sample data created with defects! Shape: {df.shape}")
        return df


    # Helper methods
    def _generate_device_usage(self, base_power, off_prob, n, hourly_pattern=None, timestamps=None):
        usage = np.random.choice([0, np.random.uniform(*base_power)], size=n, p=[off_prob, 1-off_prob])
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
        df[column] = np.clip(df[column], 0, 1)  # Keep within bounds if needed
        print(f"💧 Injected sensor drift in '{column}'")

    def _inject_intermittent_outage(self, df, column, outage_prob=0.02):
        """Simulate random outages where device stops reporting"""
        outage_mask = np.random.rand(len(df)) < outage_prob
        df.loc[outage_mask, column] = 0
        print(f"⚡ Injected intermittent outages in '{column}'")

    def _inject_missing_timestamps(self, timestamps, df, missing_prob=0.01):
        """Simulate missing rows due to logging errors"""
        missing_mask = np.random.rand(len(df)) < missing_prob
        df.drop(df[missing_mask].index, inplace=True)
        print(f"🕒 Injected missing timestamps")


    def populate_weather_and_timepoint(self, df):
        """Populate Weather and TimePoint tables"""
        cursor = self.conn.cursor()
        
        weather_data = []
        timepoint_data = []
        
        for _, row in df.iterrows():
            # Insert weather data
            weather_record = (
                row.get('temperature', 20),
                row.get('humidity', 0.5),
                row.get('pressure', 1013),
                row.get('summary', 'Clear'),
                row.get('windSpeed', 5),
                row.get('cloudCover', 0.5),
                row.get('precipIntensity', 0),
                row.get('precipProbability', 0),
                row.get('apparentTemperature', 20),
                row.get('dewPoint', 10),
                row.get('windBearing', 180),
                row.get('visibility', 10)
            )
            weather_data.append(weather_record)
        
        # Bulk insert weather data
        cursor.executemany('''
            INSERT INTO Weather (temperature, humidity, pressure, summary, wind_speed, 
                               cloud_cover, precip_intensity, precip_probability, 
                               apparent_temperature, dew_point, wind_bearing, visibility)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', weather_data)
        
        # Get weather IDs
        cursor.execute('SELECT weather_id FROM Weather ORDER BY weather_id')
        weather_ids = [row[0] for row in cursor.fetchall()]
        
        # Insert timepoint data
        for i, (_, row) in enumerate(df.iterrows()):
            timepoint_record = (
                int(row.get('time', datetime.now().timestamp())),
                row.get('House overall [kW]', 0),
                row.get('Solar [kW]', 0),
                weather_ids[i]
            )
            timepoint_data.append(timepoint_record)
        
        cursor.executemany('''
            INSERT INTO TimePoint (timestamp, house_overall_usage, solar_generation, weather_id)
            VALUES (?, ?, ?, ?)
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
    
    def run_sample_queries(self):
        """Run sample queries to demonstrate the normalized database"""
        cursor = self.conn.cursor()
        
        print("\n🔍 Sample Query Results:")
        print("=" * 50)
        
        # Query 1: Total solar generation vs consumption (last 7 days)
        print("\n1. Solar Generation vs Consumption (Last 7 Days):")
        one_week_ago = int((datetime.now() - timedelta(days=7)).timestamp())
        
        cursor.execute('''
            SELECT 
                SUM(solar_generation) as total_solar,
                SUM(house_overall_usage) as total_consumption,
                COUNT(*) as data_points
            FROM TimePoint 
            WHERE timestamp > ?
        ''', (one_week_ago,))
        
        result = cursor.fetchone()
        if result:
            solar, consumption, points = result
            print(f"   Solar Generated: {solar:.2f} kW")
            print(f"   House Consumed:  {consumption:.2f} kW")
            print(f"   Data Points:     {points}")
            if solar > 0:
                print(f"   Solar Efficiency: {(solar/consumption)*100:.1f}%")
        
        # Query 2: Top 5 energy consuming devices
        print("\n2. Top 5 Energy Consuming Devices:")
        cursor.execute('''
            SELECT 
                d.name,
                r.name as room,
                AVG(dd.power_usage) as avg_power,
                MAX(dd.power_usage) as max_power,
                COUNT(dd.data_id) as measurements
            FROM Device d
            JOIN DeviceData dd ON d.device_id = dd.device_id
            JOIN Room r ON d.room_id = r.room_id
            WHERE dd.power_usage > 0
            GROUP BY d.device_id
            ORDER BY avg_power DESC
            LIMIT 5
        ''')
        
        for i, (name, room, avg_power, max_power, measurements) in enumerate(cursor.fetchall(), 1):
            print(f"   {i}. {name} ({room})")
            print(f"      Avg: {avg_power:.3f} kW, Max: {max_power:.3f} kW, Readings: {measurements}")
        
        # Query 3: Weather impact on energy consumption
        print("\n3. Energy Consumption by Weather Condition:")
        cursor.execute('''
            SELECT 
                w.summary,
                AVG(tp.house_overall_usage) as avg_consumption,
                AVG(w.temperature) as avg_temp,
                COUNT(*) as occurrences
            FROM TimePoint tp
            JOIN Weather w ON tp.weather_id = w.weather_id
            GROUP BY w.summary
            ORDER BY avg_consumption DESC
        ''')
        
        for condition, avg_consumption, avg_temp, count in cursor.fetchall():
            print(f"   {condition:10}: {avg_consumption:.2f} kW (avg temp: {avg_temp:.1f}°C, {count} times)")
        
        # Query 4: Devices by room
        print("\n4. Device Distribution by Room:")
        cursor.execute('''
            SELECT 
                r.name as room,
                r.floor,
                COUNT(d.device_id) as device_count,
                GROUP_CONCAT(d.name, ', ') as devices
            FROM Room r
            LEFT JOIN Device d ON r.room_id = d.room_id
            GROUP BY r.room_id
            ORDER BY r.floor, r.name
        ''')
        
        for room, floor, count, devices in cursor.fetchall():
            print(f"   Floor {floor} - {room}: {count} devices")
            if devices and len(devices) < 100:  # Avoid very long device lists
                print(f"      ({devices})")
    
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
                writer.writerows(rows)         # Write data
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
    normalizer = SmartHomeDBNormalizer('smarthome_normalized.db')
    
    try:
        # Normalize the database (using sample data)
        normalizer.normalize_database()
        
        # Run sample queries
        normalizer.run_sample_queries()

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

# Additional utility functions for specific queries mentioned in your requirements

def execute_specific_queries(db_path='smarthome_normalized.db'):
    """Execute the specific queries mentioned in your requirements"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n🎯 Executing Specific Queries from Requirements:")
    print("=" * 60)
    
    # Query 2: Retrieve the power usage of Dishwasher at specific time
    print("\nQuery 2: Dishwasher power usage at specific time ID:")
    cursor.execute('''
        SELECT dd.power_usage, tp.timestamp, d.name
        FROM DeviceData dd
        JOIN Device d ON dd.device_id = d.device_id
        JOIN TimePoint tp ON dd.time_id = tp.time_id
        WHERE d.name LIKE '%Dishwasher%' AND dd.time_id <= 100
        ORDER BY dd.time_id DESC
        LIMIT 5
    ''')
    
    results = cursor.fetchall()
    if results:
        for power, timestamp, device_name in results:
            dt = datetime.fromtimestamp(timestamp)
            print(f"   {device_name}: {power:.4f} kW at {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Query 6: High power usage devices in cold weather
    print("\nQuery 6: High power usage (>0.5 kW) in cold weather (<5°C):")
    cursor.execute('''
        SELECT d.name, dd.power_usage, w.temperature, tp.timestamp
        FROM DeviceData dd
        JOIN Device d ON dd.device_id = d.device_id
        JOIN TimePoint tp ON dd.time_id = tp.time_id
        JOIN Weather w ON tp.weather_id = w.weather_id
        WHERE dd.power_usage > 0.5 AND w.temperature < 5
        ORDER BY dd.power_usage DESC
        LIMIT 10
    ''')
    
    results = cursor.fetchall()
    if results:
        for device_name, power, temp, timestamp in results:
            dt = datetime.fromtimestamp(timestamp)
            print(f"   {device_name}: {power:.3f} kW at {temp:.1f}°C ({dt.strftime('%m/%d %H:%M')})")
    
    conn.close()

if __name__ == "__main__":
    main()
    # execute_specific_queries()