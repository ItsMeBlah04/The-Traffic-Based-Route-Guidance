import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self):
        pass

    def cosine_normalize(self, df:pd.DataFrame, name:str, divide:int=None) -> pd.DataFrame:
        """
        Normalize a feature using cosine normalization.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            name (str): Name of the column to be normalized.
            divide (int): The value to divide the feature by for normalization.

        Rerturns:
            pd.DataFrame: DataFrame with cosine normalized features.
        """
        if divide is None:
            raise ValueError("Divide parameter must be provided for cosine normalization.")
        
        temp_df = df.copy()

        temp_df[f'{name}_sin'] = np.sin(2 * np.pi * temp_df[name] / divide)
        temp_df[f'{name}_cos'] = np.cos(2 * np.pi * temp_df[name] / divide)
        temp_df[f'{name}_sin'] = temp_df[f'{name}_sin'].astype('float32')
        temp_df[f'{name}_cos'] = temp_df[f'{name}_cos'].astype('float32')

        return temp_df
    
    def categorical_encoder(self, df:pd.DataFrame, name:str) -> pd.DataFrame:

        """
        Encode categorical features using Label Encoding.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            name (str): Name of the column to be encoded.

        Returns:
            pd.DataFrame: DataFrame with encoded features.
        """
        self.label_encoder = LabelEncoder()
        temp_df = df.copy()

        temp_df[name] = self.label_encoder.fit_transform(temp_df[name])
        temp_df[name] = temp_df[name].astype('float32')

        # save the label encoder for future use
        if not os.path.exists('./preprocessor'):
            os.makedirs('./preprocessor')

        if not os.path.exists('./preprocessor/label_encoders'):
            os.makedirs('./preprocessor/label_encoders')

        joblib.dump(self.label_encoder, f'./preprocessor/label_encoders/{name}_encoder.pkl')

        return temp_df
    
    def numerical_normalizer(self, df: pd.DataFrame, names: list) -> pd.DataFrame:
        """
        Normalize multiple numerical features using Min-Max Scaling and 
        save each scaler separately.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            names (list): List of column names to normalize.

        Returns:
            pd.DataFrame: DataFrame with normalized features.
        """
        temp_df = df.copy()

        if not os.path.exists('./preprocessor'):
            os.makedirs('./preprocessor')

        if not os.path.exists('./preprocessor/scalers'):
            os.makedirs('./preprocessor/scalers')

        for name in names:
            scaler = MinMaxScaler()
            temp_df[[name]] = scaler.fit_transform(temp_df[[name]])
            temp_df[name] = temp_df[name].astype('float32')

            # Save the scaler
            joblib.dump(scaler, f'./preprocessor/scalers/{name}_scaler.pkl')

        return temp_df


    def preprocess_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by normalizing and encoding features.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        print("Processing dataframe...")
        # normalize cosine for time, day of month, and weekday
        df['Time'] = df['Time'].str.extract(r'V(\d+)')[0].astype(int)
        df = self.cosine_normalize(df, 'Time', 96)
        df = df.drop(columns=['Time'])

        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfMonth'] = df['Date'].dt.day
        df = self.cosine_normalize(df, 'DayOfMonth', 31)
        df = df.drop(columns=['Date', 'DayOfMonth'])

        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }

        df['Weekday'] = df['Weekday'].map(weekday_map)
        df = self.cosine_normalize(df, 'Weekday', 7)
        df = df.drop(columns=['Weekday'])

        # encode categorical features
        df = self.categorical_encoder(df, 'Site Type')
        df = self.categorical_encoder(df, 'Location')

        # normalize numerical features
        df = self.numerical_normalizer(df, ['Latitude', 'Longitude', 'Volume', 'Location'])

        df = df[['Location', 'day_gap','Latitude', 'Longitude', 'Site Type', 'Volume', 'Time_sin', 'Time_cos', 'DayOfMonth_sin', 'DayOfMonth_cos', 'Weekday_sin', 'Weekday_cos']]

        print(f"Data info after preprocessing:\n{df.info()}")
        print(f"Data duplicates after preprocessing: {df.duplicated().sum()}")
        print(f"Data head after preprocessing:\n{df.head()}")

        return df
    
    def data_sequencer(self, df: pd.DataFrame, seq_len=24, test_ratio=0.2) -> tuple:
        """
        Create sequences of data for training, validation, and testing.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            seq_len (int): Length of the sequences.
            test_ratio (float): Ratio of the data to be used for testing.

        Returns:
            tuple: Sequences and targets for training, validation, and testing.
        """
        print("Creating sequences by location...")
        temp_df = df.copy()
        
        X_train_all, y_train_all = [], []
        X_test_all, y_test_all = [], []
        X_val_all, y_val_all = [], []

        # Group by Location
        grouped = temp_df.groupby('Location')

        for location, group in grouped:
            group = group.sort_values('day_gap').reset_index(drop=True)

            # Identify continuous time segments
            group['segment_id'] = (group['day_gap'] > 1).cumsum()

            for segment_id, segment_df in group.groupby('segment_id'):
                if len(segment_df) < seq_len + 1:
                    continue

                sequences = []
                targets = []

                for i in range(len(segment_df) - seq_len):
                    seq = segment_df.iloc[i:i+seq_len].drop(columns=['Volume', 'day_gap', 'segment_id']).values
                    target = segment_df.iloc[i+seq_len]['Volume']
                    sequences.append(seq)
                    targets.append(target)

                num_total = len(sequences)
                if num_total == 0:
                    continue

                # Split the data into training val and test sets (80% train, 10% val, 10% test)
                num_train = int(num_total * (1 - test_ratio))
                num_test = int(num_total * (test_ratio / 2))

                X_train, y_train = sequences[:num_train], targets[:num_train]
                X_test, y_test = sequences[num_train:num_train + num_test], targets[num_train:num_train + num_test]
                X_val, y_val = sequences[num_train + num_test:], targets[num_train + num_test:]

                # Append to the all data lists
                X_train_all.extend(X_train)
                y_train_all.extend(y_train)
                X_test_all.extend(X_test)
                y_test_all.extend(y_test)
                X_val_all.extend(X_val)
                y_val_all.extend(y_val)

        # Convert to NumPy arrays
        X_train = np.array(X_train_all, dtype='float32')
        y_train = np.array(y_train_all, dtype='float32')
        X_val = np.array(X_val_all, dtype='float32')
        y_val = np.array(y_val_all, dtype='float32')
        X_test = np.array(X_test_all, dtype='float32')
        y_test = np.array(y_test_all, dtype='float32')

        return X_train, y_train, X_test, y_test, X_val, y_val

    def load_data(self, file_path:str):
        """
        Load and preprocess the data from a CSV file and create sequences.

        Parameters:
            file_path (str): Path to the CSV file.
        """
        # Load the data
        df = pd.read_csv(file_path)

        # Preprocess the data
        df = self.preprocess_data(df)

        # Create sequences
        X_train, y_train, X_test, y_test, X_val, y_val = self.data_sequencer(df)

        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Shape of X_val: {X_val.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        print(f"Shape of y_test: {y_test.shape}")
        print(f"Shape of y_val: {y_val.shape}")

        if not os.path.exists('./datasets'):
            os.makedirs('./datasets')

        np.savez_compressed('./datasets/dataset.npz', 
                            X_train=X_train,
                            X_val=X_val, 
                            X_test=X_test, 
                            y_train=y_train,
                            y_val=y_val, 
                            y_test=y_test)
        
        print("Data saved to ./datasets/dataset.npz")
    
if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.load_data('./datasets/temp.csv')