import numpy as np
import pandas as pd
import math
import polars as pl
import joblib
import random

from datetime import datetime
from tools.gru_training import GRUModel
from tools.lstm_training import LSTMModel
from tools.cnn_lstm_training import CNNLSTMModel
from pathfinders.pathfinders import PathFinders

class RoutePlanner:
    def __init__(self):
        self.location_df = pl.read_csv("./datasets/node_id_to_location.csv")
        self.location_encode = pl.read_csv("./datasets/unique_locations.csv")
        self.original_data = pl.read_csv("./datasets/temp.csv").with_columns(
            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d")
        )
        self.encoders = {
            "Location": joblib.load("./preprocessor/label_encoders/Location_encoder.pkl"),
            "Site Type": joblib.load("./preprocessor/label_encoders/Site Type_encoder.pkl"),
        }
        self.normalizers = {
            "Latitude": joblib.load("./preprocessor/scalers/Latitude_scaler.pkl"),
            "Longitude": joblib.load("./preprocessor/scalers/Longitude_scaler.pkl"),
            "Location": joblib.load("./preprocessor/scalers/Location_scaler.pkl"),
            "Volume": joblib.load("./preprocessor/scalers/Volume_scaler.pkl"),
        }

    def get_model(self, model_type: str = None):
        if model_type == 'gru':
            self.model = GRUModel(model_path="./checkpoints/gru/gru_model.keras")
        elif model_type == 'lstm':
            self.model = LSTMModel(model_path="./checkpoints/lstm/lstm_model.keras")
        elif model_type == 'cnn_lstm':
            self.model = CNNLSTMModel(model_path="./checkpoints/cnnlstm/cnn_lstm_model.keras")
        else:
            raise ValueError("Unsupported model type. Choose from 'gru', 'lstm', or 'cnn_lstm'.")
        
    def location_to_id(self, location: str) -> tuple:
        location = location.strip().lower()
        matched = self.location_df.filter(pl.col("Location") == location)

        if matched.is_empty():
            raise ValueError(f"Location '{location}' not found in the dataset.")

        node_id = matched.select("Node ID").item()
        # lat = matched.select("Latitude").item()
        # lon = matched.select("Longitude").item()
        # site_type = matched.select("Site Type").item()

        return int(node_id)
    
    def id_to_location(self, node_id: int) -> str:
        """
        Given a node ID, return the corresponding location name.

        Args:
            node_id (int): Node ID to look up

        Returns:
            str: The full location name from the dataset
            longitude, latitude
        Raises:
            ValueError: If the node ID is not found
        """
        matched = self.location_df.filter(pl.col("Node ID") == node_id)

        if matched.is_empty():
            raise ValueError(f"Node ID '{node_id}' not found in the dataset.")

        location = matched.select("Location").item()
        lat = matched.select("Latitude").item()
        lon = matched.select("Longitude").item()

        return location, (lat, lon)
    
    def matched_nearest_location(self, location: str) -> str:
        """
        Given a partial location (e.g., 'auburn_rd'), return the full matching location name.

        First tries prefix match. If no match, tries suffix match (last word match).

        Args:
            location (str): Partial location (case-insensitive, underscored)

        Returns:
            str: Best-matched full location name

        Raises:
            ValueError: If no match is found
        """
        df = self.location_encode.with_columns([
            pl.col("Location").str.to_lowercase().alias("Location_lc")
        ])

        query = location.lower()

        match = df.filter(pl.col("Location_lc").str.starts_with(query))

        if match.is_empty():
            last_word = query.split("_")[-1]
            match = df.filter(pl.col("Location_lc").str.ends_with(last_word))

        if match.is_empty():
            raise ValueError(f"No location found that starts with or ends with '{location}'")

        return match[0, "Location"]

    
    def convert_date(self, date: str) -> str:
        """
        Convert any date input to 'dd/10/2006' format.

        Args:
            date (str): Original date string (any common format)

        Returns:
            str: Reformatted date with month as 10 and year as 2006
        """
        parsed_date = datetime.strptime(date, "%d/%m/%Y")  

        converted = parsed_date.replace(month=10, year=2006)

        return converted.strftime("%d/%m/%Y")
    
    def convert_time_to_vcode(self, time_str: str) -> str:
        """
        Convert a time string (HH:MM) to a 15-minute interval code like 'V32'.

        Args:
            time_str (str): Time in 'HH:MM' format (24-hour)

        Returns:
            str: Corresponding VXX code
        """
        hour, minute = map(int, time_str.split(":"))

        if minute % 15 != 0:
            raise ValueError("Time must be at a 15-minute interval (e.g., 08:00, 08:15)")

        interval = hour * 4 + (minute // 15)

        if interval < 0 or interval > 95:
            raise ValueError("Invalid time range. Must be between 00:00 and 23:45")

        return f"V{interval:02}"

    def preprocess_data_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a 24-row input sequence using preloaded encoders and normalizers.

        Args:
            df (pd.DataFrame): Raw 24-step sequence.

        Returns:
            pd.DataFrame: Preprocessed sequence ready for model input.
        """
        df = df.copy()

        df["Time"] = df["Time"].str.extract(r"V(\d+)")[0].astype(int)
        df["Time_sin"] = np.sin(2 * np.pi * df["Time"] / 96).astype("float32")
        df["Time_cos"] = np.cos(2 * np.pi * df["Time"] / 96).astype("float32")
        df = df.drop(columns=["Time"])

        df["Date"] = pd.to_datetime(df["Date"])
        df["DayOfMonth"] = df["Date"].dt.day
        df["DayOfMonth_sin"] = np.sin(2 * np.pi * df["DayOfMonth"] / 31).astype("float32")
        df["DayOfMonth_cos"] = np.cos(2 * np.pi * df["DayOfMonth"] / 31).astype("float32")
        df = df.drop(columns=["Date", "DayOfMonth"])

        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df["Weekday"] = df["Weekday"].map(weekday_map)
        df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7).astype("float32")
        df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7).astype("float32")
        df = df.drop(columns=["Weekday"])

        for col in ["Location", "Site Type"]:
            encoder = self.encoders[col]
            df[col] = encoder.transform(df[col])
            df[col] = df[col].astype("float32")

        for col in ["Latitude", "Longitude", "Location", "Volume"]:
            normalizer = self.normalizers[col]
            df[[col]] = normalizer.transform(df[[col]])
            df[col] = df[col].astype("float32")

        df = df[[
            "Location", "day_gap", "Latitude", "Longitude", "Site Type", "Volume",
            "Time_sin", "Time_cos", "DayOfMonth_sin", "DayOfMonth_cos",
            "Weekday_sin", "Weekday_cos"
        ]]
        df.drop(columns=["day_gap", "Volume"], inplace=True, errors='ignore')
        return df.reset_index(drop=True)

    def get_input_sequence(self, location: str, date_str: str, time_str: str, seq_len: int = 24) -> pd.DataFrame:
        """
        Extract and preprocess a 24-step input sequence ending before the specified time.
        If not enough history is available, randomly select a valid 24-step window from the same day.
        If the exact date is not found, fallback to the latest available date.

        Args:
            location (str): Exact full location string.
            date_str (str): Date string like '01/10/2006'.
            time_str (str): Time string like 'V36'.
            seq_len (int): Number of steps (default: 24)

        Returns:
            pd.DataFrame: Preprocessed 24-row input sequence (Pandas format).
        """
        location_data = self.original_data.filter(pl.col("Location") == location)

        if location_data.is_empty():
            raise ValueError(f"Location '{location}' not found in the dataset.")

        date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
        available_dates = location_data.select("Date").unique().to_series().to_list()

        if date_obj not in available_dates:
            date_obj = max(available_dates)
            print(f"Date '{date_str}' not found for location '{location}'. Using latest available date: {date_obj.strftime('%d/%m/%Y')}")

        filtered = location_data.filter(pl.col("Date") == date_obj).sort("Time")
        time_list = filtered.select("Time").to_series().to_list()

        if time_str not in time_list:
            raise ValueError(f"Time '{time_str}' not found in the data for {location} on {date_str}")

        end_idx = time_list.index(time_str)

        if end_idx < seq_len:
            # Not enough data before target time; fallback to a random valid segment
            total_rows = filtered.shape[0]
            if total_rows < seq_len:
                raise ValueError(f"Not enough data at all to extract a {seq_len}-step sequence at {location}")
            start_idx = random.randint(0, total_rows - seq_len)
            print(f"⚠️ Not enough history before {time_str}. Randomly selected sequence from index {start_idx} instead.")
            raw_sequence = filtered.slice(offset=start_idx, length=seq_len)
        else:
            raw_sequence = filtered.slice(offset=end_idx - seq_len, length=seq_len)

        raw_sequence_pd = raw_sequence.to_pandas()
        preprocessed_sequence = self.preprocess_data_sequence(raw_sequence_pd)

        return preprocessed_sequence

    def route_estimate(self, origin: str, destination: str, date: str, time: str, model_type: str = None, path_finder_type: str = None) -> tuple:
        
        if path_finder_type is None:
            raise ValueError("Path finder type must be specified. Choose from 'bfs', 'dfs', 'gbfs', 'df_limit', 'ucs', or 'a_star'.")
        
        if model_type is None:
            raise ValueError("Model type must be specified. Choose from 'gru', 'lstm', or 'cnn_lstm'.")
        
        node_id_origin = self.location_to_id(origin)
        node_id_destination = self.location_to_id(destination)

        paths, cost = self.get_route(node_id_origin, node_id_destination, path_finder_type)

        if not paths:
            raise ValueError("No path found between the origin and destination.")
        
        coordinates = []
        locations = []
        for node_id in paths:
            location, (lat, lon) = self.id_to_location(node_id)
            coordinates.append((lat, lon))
            locations.append(location)

        locations = [self.matched_nearest_location(loc) for loc in locations]

        date = self.convert_date(date) 
        # print(date)
        # exit()
        time_vcode = self.convert_time_to_vcode(time)    
        data_seqs = [self.get_input_sequence(loc, date, time_vcode) for loc in locations]
        # Convert to numpy array
        data_seqs = np.array([df.to_numpy() for df in data_seqs], dtype='float32')
        self.get_model(model_type)
        flows = [self.get_flow(data.reshape(1, data.shape[0], data.shape[1])) for data in data_seqs]
        total_flow = sum(flows)
        speeds = self.estimate_speed(total_flow)
        estimated_time = self.estimate_time(cost, total_flow, speeds)

        return locations, coordinates, estimated_time, cost, total_flow, speeds

    def get_route(self, node_id_origin, node_id_destination, path_finder_type="ucs") -> pd.DataFrame:
        if path_finder_type not in ["bfs", "dfs", "gbfs", "df_limit", "ucs", "a_star"]:
            raise ValueError("Unsupported path finder type. Choose from 'bfs', 'dfs', 'gbfs', 'df_limit', 'ucs', or 'a_star'.")
        
        if node_id_origin == node_id_destination:
            raise ValueError("Origin and destination cannot be the same.")
        
        path_finder = PathFinders("./datasets/graph_map/traffic_graph.txt")
        origin = node_id_origin
        goals = {node_id_destination}
        if path_finder_type == "bfs":
            goal, created, path = path_finder.bfs(origin, goals)
        elif path_finder_type == "dfs":
            goal, created, path = path_finder.dfs(origin, goals)
        elif path_finder_type == "gbfs":
            goal, created, path = path_finder.gbfs(origin, goals)
        elif path_finder_type == "df_limit":
            goal, created, path = path_finder.df_limited(origin, goals)
        elif path_finder_type == "ucs":
            goal, created, path = path_finder.uniform_cost_search(origin, goals)
        elif path_finder_type == "a_star":
            goal, created, path = path_finder.astar(origin, goals)

        cost = path_finder.calculate_path_cost(path)

        return path, cost

    def estimate_speed(self, flow: float) -> float:
        a = -1.4648375
        b = 93.75
        c = -flow
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("No real solution for speed")
        
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        speeds = sorted([root1, root2])
        return speeds  # [congested, free-flow]
    
    def estimate_time(self, distance_km, flow, speeds):
        if flow <= 351:
            speed = 60.0  
        else:
            speed = speeds[0]
        return (distance_km / speed) * 60  
    
    def get_flow(self, data: np.ndarray) -> float:
        """
        Predict flow using the loaded model and denormalize the result.

        Args:
            data (np.ndarray): Input of shape [batch_size, 24, 10]

        Returns:
            float: Denormalized flow prediction
        """
        # Check input shape
        if data.ndim != 3 or data.shape[1] != 24 or data.shape[2] != 10:
            raise ValueError("Data must be of shape [batch_size, 24, 10]")

        # Predict normalized flow
        normalized_flow = self.model.predict(data)

        # Ensure it's 2D shape before inverse_transform (e.g., [[0.42]])
        if normalized_flow.ndim == 2:
            flow = normalized_flow[0, 0]
        else:
            flow = normalized_flow[0]

        # Denormalize using the saved scaler
        denorm_flow = self.normalizers["Volume"].inverse_transform([[flow]])[0][0]
        return denorm_flow