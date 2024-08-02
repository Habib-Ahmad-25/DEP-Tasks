import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv(r'C:\Users\habib\Desktop\DEP\Archive\All_perth_310121.csv')  # replace with your dataset path

# Inspect the dataset to understand its structure
print(df.head())
print(df.columns)

# Handling missing values (consider imputation for a better approach)
df = df.dropna()

# Splitting features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# One-hot encoding for categorical features
categorical_features = ['ADDRESS']  # replace with your actual categorical feature names
numerical_features = ['LAND_AREA', 'BEDROOMS']  # replace with your actual numerical feature names

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)  # Set sparse=False
    ],
    remainder='passthrough'  # Ensures all other columns are passed through
)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Training the model
pipeline.fit(X_train, y_train)

# Predicting and evaluating on test data
y_pred = pipeline.predict(X_test)
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Predicting the price of a new house
new_data = pd.DataFrame({
    'ADDRESS': ['1 Acorn Place'],  # Replace with actual categorical value
    'LAND_AREA': [500],           # Replace with actual numerical value
    'BEDROOMS': [3]               # Replace with actual numerical value
})

new_pred = pipeline.predict(new_data)
print('Predicted House Price:', new_pred[0])
