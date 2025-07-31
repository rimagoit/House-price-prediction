# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Load the Dataset
# -----------------------------
# Download dataset from Kaggle: "House Prices - Advanced Regression Techniques"
# Example CSV file: 'train.csv'
df = pd.read_csv('train.csv')

# Display the dataset structure
print(df.head())

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
# Selecting relevant columns
columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
data = df[columns]

# Handle missing values (if any)
data = data.fillna(0)

# Split into features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 4: Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate the Model
# -----------------------------
y_pred = model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# -----------------------------
# Step 6: Visualization
# -----------------------------
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

# -----------------------------
# Optional: Show feature importance (coefficients)
# -----------------------------
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Coefficients:")
print(coefficients.sort_values(by='Coefficient', ascending=False))
