import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay

class CNNLSTMModel:
    def __init__(
        self,
        input_shape=None,
        model_path=None,
        conv_filters=[64],
        kernel_sizes=[3],
        pool_sizes=[2],
        lstm_units=[64, 32],
        dropout_rates=[0.3],
        learning_rate=0.001,
        loss='mse',
        metrics=['mae'],
        patience=5,
        use_cosine_decay=False,
        batch_size=128,
        epochs=10,
    ):
        """
        Initializes the CNNLSTMModel instance.

        Parameters:
            input_shape (tuple): Shape of input data (timesteps, features).
            model_path (str): Path to a saved model (optional).
            conv_filters (list): List of filter counts for each Conv1D layer.
            kernel_sizes (list): List of kernel sizes for each Conv1D layer.
            pool_sizes (list): List of pooling sizes after each Conv1D layer.
            lstm_units (list): List of LSTM units for each LSTM layer.
            dropout_rates (list): List of dropout rates after Conv/LSTM layers.
            learning_rate (float): Learning rate for the Adam optimizer.
            loss (str): Loss function.
            metrics (list): List of evaluation metrics.
            patience (int): Early stopping patience.
            use_cosine_decay (bool): Whether to apply cosine learning rate decay.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
        """
        self.input_shape = input_shape
        self.model_path = model_path
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.patience = patience
        self.use_cosine_decay = use_cosine_decay
        self.batch_size = batch_size
        self.epochs = epochs

        if model_path:
            self.model = load_model(model_path)
        elif input_shape:
            self.model = self.build_model()
        else:
            raise ValueError("You must provide either input_shape or model_path")

    def build_model(self):
        """
        Build the CNN-LSTM model architecture.

        Returns:
            model (Sequential): Compiled Keras model.
        """
        model = Sequential()
        
        # Add Conv1D + Pooling layers
        for i in range(len(self.conv_filters)):
            if i == 0:
                model.add(Conv1D(filters=self.conv_filters[i], kernel_size=self.kernel_sizes[i],
                                 activation='relu', input_shape=self.input_shape))
            else:
                model.add(Conv1D(filters=self.conv_filters[i], kernel_size=self.kernel_sizes[i],
                                 activation='relu'))
            model.add(MaxPooling1D(pool_size=self.pool_sizes[i]))
            if i < len(self.dropout_rates):
                model.add(Dropout(self.dropout_rates[i]))

        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_seq = i < len(self.lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_seq))
            if i + len(self.conv_filters) < len(self.dropout_rates):
                model.add(Dropout(self.dropout_rates[i + len(self.conv_filters)]))

        # Output layer
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss, metrics=self.metrics)
        return model

    def train(self, x_train, y_train, x_val, y_val, vis=False):
        """
        Train the CNN-LSTM model.

        Parameters:
            x_train (array): Training input data.
            y_train (array): Training target values.
            x_val (array): Validation input data.
            y_val (array): Validation target values.

        Returns:
            history (History): Training history object.
        """
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )

        if self.use_cosine_decay:
            CosineDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=(self.epochs * len(x_train)) // self.batch_size,
                alpha=0.0
            )

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        if vis:
            self.visualize_training_log(history)

        return history

    def evaluate(self, x_test, y_test):
        """
        Evaluate the trained model.

        Parameters:
            x_test (array): Test input data.
            y_test (array): Ground truth values.

        Returns:
            mse (float): Mean Squared Error.
            mae (float): Mean Absolute Error.
            r2 (float): R-squared Score.
        """
        predictions = self.model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f'MSE: {mse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R2: {r2:.4f}')

        return predictions, (mse, mae, r2)

    def save(self, path):
        """
        Save the model to the specified path.

        Parameters:
            path (str): Destination file path.
        """
        self.model.save(path)

    def predict(self, x_input):
        """
        Generate predictions for input data.

        Parameters:
            x_input (array): Input data.

        Returns:
            predictions (array): Predicted output.
        """
        return self.model.predict(x_input)
    
    def visualize_training_log(self, history):
        """
        Visualizes training and validation loss and metrics from a Keras History object.

        Parameters:
            history (History): The Keras History object returned by model.fit().
        """
        if not history or not hasattr(history, 'history'):
            raise ValueError("Invalid history object. Make sure it's from model.fit()")

        history_dict = history.history

        # Detect all metric names (excluding val_ versions)
        metrics = [m for m in history_dict.keys() if not m.startswith("val_")]

        # Plot each metric and its validation counterpart
        for metric in metrics:
            plt.figure(figsize=(8, 4))
            plt.plot(history_dict[metric], label=f"Training {metric}")
            val_key = f"val_{metric}"
            if val_key in history_dict:
                plt.plot(history_dict[val_key], label=f"Validation {metric}")
            plt.title(f"{metric.capitalize()} over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def load(path):
        """
        Load a saved CNN-LSTM model.

        Parameters:
            path (str): Path to the saved model.

        Returns:
            CNNLSTMModel: An instance with the loaded model.
        """
        return CNNLSTMModel(model_path=path)