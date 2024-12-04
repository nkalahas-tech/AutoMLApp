import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Instructions
st.title("AutoML Application with Preprocessing")
st.write("Upload your dataset, and let the application run machine learning models for you!")

# Dataset Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Select Target Variable
    target_column = st.selectbox("Select the Target Column", data.columns)

    # Preprocessing
    st.write("Preprocessing Data...")



    # Preprocessing function for feature columns only
    def preprocess_features(df, target_column):
        df = df.drop(columns=[target_column])  # Exclude the target column from feature processing
        for col in df.columns:
            if df[col].dtype == 'object':  # Categorical or string columns
                try:
                    # Attempt to convert to datetime, then to numeric timestamp
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = df[col].astype('int64') // 10 ** 9  # Convert to seconds since epoch
                except Exception:
                    # Encode categorical text
                    df[col] = df[col].astype('category').cat.codes

            # Handle remaining non-numeric or missing values
            if df[col].isnull().any() or df[col].dtype == 'object':
                df[col].fillna(-1, inplace=True)  # Fill missing categorical values with -1

        # Ensure numeric columns have no missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df


    # Apply preprocessing to features
    X = preprocess_features(data, target_column)

    # Process the target column separately
    y = data[target_column]

    # Validate target column
    if y.nunique() <= 1:
        st.error("Target column must contain at least two classes for classification.")
        raise ValueError("Target column contains fewer than two unique classes.")

    # Ensure target column is properly encoded if categorical
    if y.dtype == 'object' or str(y.dtype) == 'category':
        y = y.astype('category').cat.codes

    # X is now safe to use with the models
    # Split Data into Features and Target

    # Automatically detect the problem type
    problem_type = "classification" if y.nunique() <= 10 else "regression"
    st.write(f"Detected Problem Type: {problem_type.capitalize()}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run Models
    st.write("Running Models...")
    results = {}

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(max_iter=500)
        }
    else:  # Regression
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Neural Network": MLPRegressor(max_iter=500)
        }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        if problem_type == "classification":
            # Classification: Use accuracy as the metric
            from sklearn.metrics import accuracy_score

            score = accuracy_score(y_test, predictions)
        else:
            # Regression: Use R² as the metric
            from sklearn.metrics import r2_score

            score = r2_score(y_test, predictions)
        results[model_name] = score

    # Display Results
    st.write("Model Performance:")
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Score"])
    st.table(results_df)

    # Visualization
    if "Random Forest" in models:
        rf_model = models["Random Forest"]
        if hasattr(rf_model, "feature_importances_"):
            feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
            st.write("Feature Importance (Random Forest):")
            plt.figure(figsize=(10, 6))
            feature_importance.nlargest(10).plot(kind="barh", color="skyblue")
            plt.title("Top 10 Feature Importance")
            st.pyplot(plt)

