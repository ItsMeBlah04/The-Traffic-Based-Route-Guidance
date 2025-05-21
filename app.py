import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained model
lstm_model = joblib.load("LSTM.pkl")

# Sidebar
st.sidebar.title("TBRGS - Traffic Prediction")
option = st.sidebar.selectbox("Select Action", ["Predict", "Visualize Data", "Model Comparison"])

# Main title
st.title("Traffic-Based Route Guidance System (TBRGS)")

# Load dataset from .npz and convert to serializable dictionary
@st.cache_data
def load_npz_data(path):
    npzfile = np.load(path)
    return {key: npzfile[key] for key in npzfile.files}

data = load_npz_data("dataset.npz")

# Extract training data
X_train = data["X_train"]
y_train = data["y_train"]

# Assume SCATS_Number is in column 0
scats_list = np.unique(X_train[:, 0])

st.write("Available arrays in dataset:", list(data.keys()))

# ------------------------- Prediction -----------------------------
if option == "Predict":
    st.subheader("Predict Traffic Volume")

    scats_site = st.selectbox("Select SCATS Site", sorted(scats_list))
    hour = st.slider("Hour", 0, 23, 12)
    minute = st.slider("Minute", 0, 59, 0)
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

    # Create input DataFrame
    input_data = pd.DataFrame([[scats_site, hour, minute, day_map[day_of_week]]],
                              columns=["SCATS_Number", "Hour", "Minute", "DayOfWeek"])

    model_choice = st.radio("Select model", ["LSTM"])

    if st.button("Predict"):
        if model_choice == "LSTM":
            prediction = lstm_model.predict(input_data)
            st.success(f"Predicted Traffic Volume: {prediction[0]:.2f} vehicles")

# ----------------------- Visualization ----------------------------
elif option == "Visualize Data":
    st.subheader("Traffic Data Overview")

    X_train = data["X_train"]
    y_train = data["y_train"]

    # Flatten the last timestep
    X_flat = X_train[:, -1, :]
    df_all = pd.DataFrame(X_flat, columns=[f"F{i}" for i in range(X_flat.shape[1])])
    df_all["Volume"] = y_train

    # Assume SCATS site number is in F0 (adjust if needed)
    scats_list = np.unique(df_all["F0"])
    site = st.selectbox("Choose SCATS Site", sorted(scats_list))

    df_site = df_all[df_all["F0"] == site]

    fig, ax = plt.subplots()
    ax.plot(df_site["F1"], df_site["Volume"], marker='o', linestyle='-')  # F1 might be Hour
    ax.set_title(f"Traffic Volume at SCATS Site {int(site)}")
    ax.set_xlabel("Hour (assumed: F1)")
    ax.set_ylabel("Volume")
    st.pyplot(fig)


# ---------------------- Model Comparison --------------------------
elif option == "Model Comparison":
    st.subheader("Model Performance Comparison")
    results = {
        "Model": ["LSTM"],
        "MAE": [15.3],
        "RMSE": [20.2]
    }
    st.dataframe(pd.DataFrame(results))
