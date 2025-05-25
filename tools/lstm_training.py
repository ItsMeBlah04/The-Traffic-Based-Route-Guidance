import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay

class LSTMModel:
    def __init__(
        self,
        input_shape=None,
        model_path=None,
        gru_units=[64, 32, 16],
        dropout_rates=[0.3, 0.2],
        learning_rate=0.001,
        loss='mse',
        metrics=['mae'],
        patience=5,
        use_cosine_decay=False,
        batch_size=128,
        epochs=10,
    ):
        """
        Initializes the LSTMModel instance.

        Args:
            input_shape (tuple): Shape of input data (timesteps, features).
            model_path (str): Path to load a pre-trained model (overrides building).
            gru_units (list): List of GRU layer sizes.
            dropout_rates (list): List of dropout rates between GRU layers.
            learning_rate (float): Initial learning rate for Adam.
            loss (str): Loss function.
            metrics (list): List of metrics to monitor.
            patience (int): Early stopping patience.
            use_cosine_decay (bool): Whether to apply cosine LR decay (not attached by default).
            batch_size (int): Batch size for training.
            epochs (int): Default number of training epochs.
        """
        self.input_shape = input_shape
        self.model_path = model_path
        self.gru_units = gru_units
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
            raise ValueError("You must provide either input_shape (to build) or model_path (to load)")

    def build_model(self):
        """
        Build a GRU-based model with configurable architecture.

        Returns:
            model: Compiled Keras model.
        """
        model = Sequential()
        for i, units in enumerate(self.gru_units):
            return_seq = i < len(self.gru_units) - 1
            if i == 0:
                model.add(GRU(units, return_sequences=return_seq, input_shape=self.input_shape))
            else:
                model.add(GRU(units, return_sequences=return_seq))
            if i < len(self.dropout_rates):
                model.add(Dropout(self.dropout_rates[i]))

        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss, metrics=self.metrics)
        return model

    def train(self, x_train, y_train, x_val, y_val, vis=False):
        """
        Train the model with early stopping and optional cosine decay.

        Parameters:
            x_train (array): Training data.
            y_train (array): Training labels.
            x_val (array): Validation data.
            y_val (array): Validation labels.

        Returns:
            history: Training history.
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
        Evaluate the model on test data.

        Parameters:
            x_test (array): Test data.
            y_test (array): Test labels.

        Returns:
            mse (float): Mean Squared Error.
            mae (float): Mean Absolute Error.
            r2 (float): R-squared score.
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
            path (str): Path to save the model.
        """
        self.model.save(path)

    def predict(self, x_input):
        """
        Make predictions on new data.

        Parameters:
            x_input (array): Input data for prediction.
        Returns:
            predictions (array): Model predictions.
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
        Load a pre-trained model from the specified path.

        Parameters:
            path (str): Path to the pre-trained model.
        Returns:
            LSTMModel: Instance of LSTMModel with the loaded model.
        """
        return LSTMModel(model_path=path)