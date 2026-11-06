
# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# STEP 2: LOAD THE DATASET

data_path = Path("dataset/lahore_aqi_2019_to_2023.csv")

df = pd.read_csv(data_path)

print("Dataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# STEP 3: PREPROCESS THE DATA

# Remove duplicate rows
df = df.drop_duplicates()

# Convert the target column to numeric values
df["aqi_pm2.5"] = pd.to_numeric(
    df["aqi_pm2.5"], errors="coerce"
)

# Remove rows where AQI is missing
df = df.dropna(subset=["aqi_pm2.5"])

# Select environmental features.
# The date column is excluded from model inputs.
X = df.drop(columns=["aqi_pm2.5", "date"])

# Target variable
y = df["aqi_pm2.5"]

# Convert all feature columns to numeric values
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing feature values using column medians
X = X.fillna(X.median())

# Keep feature names for the Streamlit application
feature_names = list(X.columns)

print("\nFeatures used:")
print(feature_names)


# STEP 4: SPLIT THE DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 5: TRAIN THE MODEL

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nAQI Model Training Completed!")


# STEP 6: PREDICT AND EVALUATE

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")


# STEP 7: SAVE THE MODEL AND FEATURE NAMES

joblib.dump(model, "model.joblib")

joblib.dump(
    feature_names,
    "feature_names.joblib"
)

# Save median values for the application defaults
joblib.dump(
    X.median().to_dict(),
    "feature_medians.joblib"
)

print("\nModel saved as model.joblib")
print("Feature names saved as feature_names.joblib")
print("Feature medians saved as feature_medians.joblib")

