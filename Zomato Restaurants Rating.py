import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
@st.cache_data
def load_data():
    zomato = pd.read_csv("/Users/mohitr/Downloads/zomato.csv")
    Zomato = zomato.drop(['url', 'address', 'phone', 'dish_liked', 'reviews_list', 'menu_item'], axis=1)
    Zomato = Zomato[Zomato.rate != 'NEW']
    Zomato = Zomato[Zomato.rate != '-']
    Zomato['rate'] = Zomato['rate'].apply(lambda x: x.replace('/5', '') if isinstance(x, str) else x).str.strip().astype(float)
    Zomato['rate'] = Zomato['rate'].fillna(Zomato['rate'].median())
    Zomato['location'] = Zomato['location'].fillna(Zomato['location'].mode().iloc[0])
    Zomato['rest_type'] = Zomato['rest_type'].fillna(Zomato['rest_type'].mode().iloc[0])
    Zomato['cuisines'] = Zomato['cuisines'].fillna(Zomato['cuisines'].mode().iloc[0])
    Zomato['approx_cost(for two people)'] = Zomato['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', '')).astype(float)
    Zomato['approx_cost(for two people)'] = Zomato['approx_cost(for two people)'].fillna(Zomato['approx_cost(for two people)'].median())
    LE = LabelEncoder()
    Zomato['online_order'] = LE.fit_transform(Zomato['online_order'])
    Zomato['book_table'] = LE.fit_transform(Zomato['book_table'])
    Zomato['location'] = LE.fit_transform(Zomato['location'])
    Zomato['rest_type'] = LE.fit_transform(Zomato['rest_type'])
    Zomato['cuisines'] = LE.fit_transform(Zomato['cuisines'])
    Zomato['listed_in(type)'] = LE.fit_transform(Zomato['listed_in(type)'])
    Zomato['listed_in(city)'] = LE.fit_transform(Zomato['listed_in(city)'])
    X = Zomato.drop(columns=['name', 'rate', 'listed_in(city)'], axis=1)
    Y = Zomato['rate']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, stratify=Y, random_state=2)
    return X_train, X_test, Y_train, Y_test, X.columns

def eda_process(X_train):
    st.header("Data Analysis")
    
    st.subheader("Data Overview")
    st.write("Training data shape:", X_train.shape)

    st.subheader("First few rows of the dataset")
    st.write(X_train.head())
    
    st.subheader("Statistical Summary")
    st.write(X_train.describe())

    st.subheader("Correlation Heatmap")
    # Create a figure and axis for the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = X_train.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)  # Pass the figure to st.pyplot()

# Create Streamlit app
st.title('Zomato Restaurant Rating Prediction')

menu = st.sidebar.selectbox("Choose an option", ["Data Analysis", "Model Training", "Make Prediction"])

X_train, X_test, Y_train, Y_test, feature_names = load_data()

if menu == "Data Analysis":
    eda_process(X_train)

elif menu == "Model Training":
    st.header("Model Training")
    st.text("Training models and evaluating performance...")
    
    # Train Linear Regression model
    LinR = LinearRegression()
    LinR.fit(X_train, Y_train)
    prediction1 = LinR.predict(X_test)
    r2_linr = r2_score(Y_test, prediction1)
    st.write(f"Linear Regression R^2 Score: {r2_linr}")

    # Train XGBoost model
    xgbr = XGBRegressor(boost='gblinear', n_estimators=100, learning_rate=0.3, max_depth=10)
    xgbr.fit(X_train, Y_train)
    prediction2 = xgbr.predict(X_test)
    r2_xgbr = r2_score(Y_test, prediction2)
    st.write(f"XGBoost R^2 Score: {r2_xgbr}")

    # Train Random Forest model
    RFR = RandomForestRegressor(n_estimators=100, random_state=80, min_samples_split=2, min_samples_leaf=0.001)
    RFR.fit(X_train, Y_train)
    prediction3 = RFR.predict(X_test)
    r2_rfr = r2_score(Y_test, prediction3)
    st.write(f"Random Forest R^2 Score: {r2_rfr}")

elif menu == "Make Prediction":
    st.header("Make Prediction")
    st.text("Enter the details for prediction:")
    
    # Collect user input
    online_order = st.selectbox("Online Order", [1, 0])
    book_table = st.selectbox("Book Table", [1, 0])
    votes = st.number_input("Votes", min_value=0)
    location = st.number_input("Location")
    rest_type = st.number_input("Restaurant Type")
    cuisines = st.number_input("Cuisines")
    cost_2_people = st.number_input("Approx Cost (for two people)")
    type = st.number_input("Type")
    
    if st.button("Predict"):
        new_data = pd.DataFrame([[online_order, book_table, votes, location, rest_type, cuisines, cost_2_people, type]], 
                                columns=feature_names)
        
        # Ensure the new data columns match the training columns
        new_data = new_data.reindex(columns=feature_names, fill_value=0)
        
        # Load the model
        xgbr = XGBRegressor(boost='gblinear', n_estimators=100, learning_rate=0.3, max_depth=10)
        xgbr.fit(X_train, Y_train)
        new_pred = xgbr.predict(new_data)
        st.write(f"The predicted rating is: {new_pred[0]}")
