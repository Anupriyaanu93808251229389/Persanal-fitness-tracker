import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Function for login
def login():
    st.markdown(
        """
        <style>
            .main {
                background-color: #f0f2f6;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
            }
            .login-title {
                color: #2C3E50;
                font-size: 30px;
                font-weight: bold;
            }
            .login-box {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            }
            .login-button {
                background-color: #28a745;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
            }
            .login-button:hover {
                background-color: #218838;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main"><div class="login-box">', unsafe_allow_html=True)
    st.markdown('<p class="login-title">Login Page</p>', unsafe_allow_html=True)

    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")

    if st.markdown('<button class="login-button">Login</button>', unsafe_allow_html=True):
        if username and password:
            st.success(f"Welcome, {username}!")
            return True
        else:
            st.error("Please enter both username and password")
            return False
    st.markdown("</div></div>", unsafe_allow_html=True)

# Main function
def main_app():
    st.title("Personal Fitness Tracker")
    st.write("Predict your calories burned based on your health parameters.")

    st.sidebar.header("User Input Parameters:")
    
    def user_input_features():
        age = st.sidebar.slider("Age", 10, 100, 30)
        bmi = st.sidebar.slider("BMI", 15, 40, 20)
        duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
        heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
        body_temp = st.sidebar.slider("Body Temperature (C)", 36, 42, 38)
        gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
        
        gender = 1 if gender_button == "Male" else 0
        
        data_model = {
            "Age": age,
            "BMI": bmi,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
            "Gender_male": gender
        }
        
        return pd.DataFrame(data_model, index=[0])

    df = user_input_features()
    
    st.write("---")
    st.header("Your Parameters:")
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    st.write(df)

    # Load dataset
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    # Merge datasets
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    # Train-test split
    train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    for data in [train_data, test_data]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = round(data["BMI"], 2)

    # Prepare training data
    train_data = train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    test_data = test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)

    X_train = train_data.drop("Calories", axis=1)
    y_train = train_data["Calories"]

    X_test = test_data.drop("Calories", axis=1)
    y_test = test_data["Calories"]

    # Train model
    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    model.fit(X_train, y_train)

    # Align prediction data with training data
    df = df.reindex(columns=X_train.columns, fill_value=0)
    
    # Prediction
    prediction = model.predict(df)

    st.write("---")
    st.header("Prediction:")
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

    st.write(f"Predicted Calories Burned: **{round(prediction[0], 2)} kilocalories**")

    st.write("---")
    st.header("Similar Results:")
    similar_data = exercise_df[
        (exercise_df["Calories"] >= prediction[0] - 10) & 
        (exercise_df["Calories"] <= prediction[0] + 10)
    ]
    st.write(similar_data.sample(5))

# Run the application
if __name__ == "__main__":
    if login():
        main_app()
