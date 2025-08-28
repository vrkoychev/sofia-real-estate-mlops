import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

current_year = datetime.now().year

# Load cleaned dataset
df = pd.read_csv("sofia-real-estate-mlops/datasets/clean_data.csv")

# --- Feature Engineering ---
df['age'] = current_year - df['year_built']

# Drop columns we don’t want as raw features
df_features = df.drop(columns=['price', 'year_built'])

# Target
y = df['price']

# --- Identify categorical and numeric columns ---
categorical_cols = ['location', 'property_type', 'construction_type', 'floor_type']
numeric_cols = ['size', 'floor_number', 'age']

# Column transformer: one-hot for categorical, standard scaling for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# Split train/test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df_features, y, test_size=0.2, random_state=42
)

# Fit preprocessor on training data
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

import joblib
# Save preprocessor and processed data
joblib.dump(preprocessor, "sofia-real-estate-mlops/datasets/processed/preprocessor.pkl")
joblib.dump((X_train, X_test, y_train, y_test), "sofia-real-estate-mlops/datasets/processed/processed_data.pkl")

print("✅ Features prepared and saved for modeling.")
