import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# Load and preprocess the dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv("/Users/mohitr/Downloads/car_insurance_claim.csv")
    return data

def preprocess_data(df):
    # Drop repeating and useless columns
    df = df.drop(['ID', 'BIRTH', 'OCCUPATION', 'CAR_TYPE', 'CLAIM_FLAG'], axis=1)
    # Convert categorical variables to numerical values
    df = df.replace(['No', 'z_No', 'no', 'z_F', 'Private', 'z_Highly Rural/ Rural'], [0, 0, 0, 0, 0, 0])
    df = df.replace(['Yes', 'yes', 'M', 'Commercial', 'Highly Urban/ Urban'], [1, 1, 1, 1, 1])
    df = df.replace(['z_High School', '<High School', 'Bachelors', 'Masters', 'PhD'], [0, 0, 1, 2, 3])
    # Convert monetary columns to float
    monetary_cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
    for col in monetary_cols:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    # Impute missing values
    df['AGE'].fillna(df['AGE'].mean(), inplace=True)
    df['YOJ'].fillna(df['YOJ'].mean(), inplace=True)
    df['INCOME'].fillna(df['INCOME'].median(), inplace=True)
    df['HOME_VAL'].fillna(df['HOME_VAL'].median(), inplace=True)
    df['CAR_AGE'].fillna(df['CAR_AGE'].median(), inplace=True)
    return df

def corrplt(df):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn", ax=ax)
    st.pyplot(fig)

def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['CLM_AMT'], bins=20, kde=True, color='#607c8e', ax=ax)
    ax.set_title('Distribution of Claim Amount')
    ax.set_xlabel('Claim Amount')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Load the dataset
file_path = 'car_insurance_claim.csv'
data = load_data(file_path)

# Streamlit App
st.title("Car Insurance Claim Prediction EDA & Modeling")

# Sidebar for navigation
st.sidebar.title("Menu")
option = st.sidebar.selectbox("Select an option", ["Show Data", "Numerical Column Distribution", "Correlation Heatmap", "Build Model", "Make Predictions"])

if option == "Show Data":
    st.header("Dataset")
    st.write(data.head())

elif option == "Numerical Column Distribution":
    st.header("Numerical Column Distribution")
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_col = st.sidebar.selectbox("Select a numerical column", num_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[selected_col], bins=20, kde=True, color='#607c8e', ax=ax)
    ax.set_title(f'Distribution of {selected_col}')
    ax.set_xlabel(selected_col)
    ax.set_ylabel('Count')
    st.pyplot(fig)

elif option == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    df = preprocess_data(data)
    corrplt(df)

elif option == "Build Model":
    st.header("Build a Model")
    df = preprocess_data(data)

    # Feature selection
    feature_cols = st.multiselect("Select feature columns", df.columns.tolist(), default=['AGE', 'INCOME', 'HOME_VAL', 'BLUEBOOK', 'CAR_AGE', 'CLM_FREQ'])
    target_col = 'CLM_AMT'

    model_type = st.selectbox("Select Model Type", ["Linear Regression", "Decision Tree", "SVM"])

    if feature_cols and st.button("Train Model"):
        X = df[feature_cols]
        y = df[target_col]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_type == "SVM":
            # Convert regression target to binary classification for SVM
            y_train = (y_train > 0).astype(int)
            y_test = (y_test > 0).astype(int)
            model = SVC(kernel="linear")

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        if model_type in ["Linear Regression", "Decision Tree"]:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R-Squared: {r2}")

            # Plot predictions vs actual
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
            st.pyplot(fig)

        elif model_type == "SVM":
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy * 100}%")
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # Save predictions to CSV
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        predictions_csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Predictions", data=predictions_csv, file_name='predictions.csv', mime='text/csv')
    else:
        st.write("Please select at least one feature column.")

elif option == "Make Predictions":
    st.header("Make Predictions")
    df = preprocess_data(data)

    # Feature selection for prediction input
    feature_cols = st.multiselect("Select feature columns for prediction", df.columns.tolist(), default=['AGE', 'INCOME', 'HOME_VAL', 'BLUEBOOK', 'CAR_AGE', 'CLM_FREQ'])

    if feature_cols:
        input_data = []
        for col in feature_cols:
            value = st.number_input(f"Input value for {col}", value=float(df[col].mean()))
            input_data.append(value)

        input_data = np.array(input_data).reshape(1, -1)
        model_type = st.selectbox("Select Model Type for prediction", ["Linear Regression", "Decision Tree", "SVM"])

        if st.button("Predict"):
            # Train the model with all data
            X = df[feature_cols]
            y = df['CLM_AMT']

            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_type == "SVM":
                y = (y > 0).astype(int)
                model = SVC(kernel="linear")

            model.fit(X, y)

            prediction = model.predict(input_data)

            if model_type == "SVM":
                prediction = "Claim" if prediction[0] == 1 else "No Claim"
            else:
                prediction = prediction[0]

            st.write(f"Prediction: {prediction}")
