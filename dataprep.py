import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("/Users/mohitr/Downloads/car_insurance_claim.csv")
# EDA and Data Analysis
print("**Data Information:**")
print(data.info())
print("\n**Descriptive Statistics:**")
print(data.describe())
print("\n**Missing Values:**")
print(data.isnull().sum())

# Visualize the distribution of CLM_AMT
plt.figure(figsize=(8, 6))
sns.histplot(data["CLM_AMT"], kde=True)
plt.title("Distribution of CLM_AMT")
plt.xlabel("Claim Amount")
plt.ylabel("Density")
plt.show()

# Scatterplot of CLM_AMT vs INSR_CAT
plt.figure(figsize=(8, 6))
sns.scatterplot(x="INSR_CAT", y="CLM_AMT", data=data)
plt.title("Relationship between CLM_AMT and INSR_CAT")
plt.xlabel("INSR_CAT")
plt.ylabel("CLM_AMT")
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Handling Missing Values
# (Implement your chosen method for handling missing values)

# Outlier Treatment
# (Implement your chosen method for outlier treatment)

# Handling Skewed Variables
data["CLM_AMT"] = np.log1p(data["CLM_AMT"])  # Example: log transformation for CLM_AMT

# Feature Scaling
scaler = StandardScaler()
numerical_features = ["CLM_AMT", "AGE", "YEAR", "TRV_MI"]
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Encoding Categorical Variables
categorical_features = ["INSR_CAT", "DRVR_RAT", "CAR_TYPE", "OCC_CODE"]
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
encoded_data.columns = encoder.get_feature_names_out(categorical_features)
data = pd.concat([data.drop(categorical_features, axis=1), encoded_data], axis=1)

# Feature Selection
# (Implement your chosen feature selection technique)

# Model Selection and Training
X = data.drop("CLM_AMT", axis=1)
y = data["CLM_AMT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n**Model: {name}**")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

# Model Tuning and Improvement
# (Implement your chosen model tuning technique)

# Model Interpretation
# (Analyze model coefficients or feature importance)

# Conclusion
# (Summarize findings and discuss model performance)