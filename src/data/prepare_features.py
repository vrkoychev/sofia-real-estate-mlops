import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# --- Constants ---
RAW_DATA_PATH = "sofia-real-estate-mlops/datasets/clean_data.csv"
PREPROCESSOR_PATH = "sofia-real-estate-mlops/datasets/processed/preprocessor.pkl"
PROCESSED_DATA_PATH = "sofia-real-estate-mlops/datasets/processed/processed_data.pkl"

# Load cleaned dataset
df = pd.read_csv(RAW_DATA_PATH)

# --- Drop rows where target 'price' is NaN ---
df = df.dropna(subset=['price'])

# --- Feature Engineering ---
current_year = datetime.now().year
df['age'] = current_year - df['year_built']

# Drop columns not needed as raw features
df_features = df.drop(columns=['price', 'year_built'])

# --- Handle missing values ---
numeric_cols = ['size', 'floor_number', 'age']
categorical_cols = ['location', 'property_type', 'construction_type', 'floor_type']

# Fill numeric NaNs with median
df_features[numeric_cols] = df_features[numeric_cols].fillna(df_features[numeric_cols].median())

# Fill categorical NaNs with 'Unknown'
df_features[categorical_cols] = df_features[categorical_cols].fillna('Unknown')

# --- Handle outliers using IQR (cap at bounds) ---
numeric_cols_with_price = numeric_cols + ['price']
for col in numeric_cols_with_price:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Cap values at lower and upper bounds
    if col in df_features.columns:
        df_features[col] = df_features[col].clip(lower=lower, upper=upper)
    else:
        # Price is not in features, cap directly in target
        df[col] = df[col].clip(lower=lower, upper=upper)

# --- Target ---
y = df['price']

# --- Column transformer: One-hot encoding for categorical, Standard scaling for numeric ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# --- Split train/test ---
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df_features, y, test_size=0.2, random_state=42
)

# Fit preprocessor on training data
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# --- Save preprocessor and processed data ---
joblib.dump(preprocessor, PREPROCESSOR_PATH)
joblib.dump((X_train, X_test, y_train, y_test), PROCESSED_DATA_PATH)

print("✅ Features prepared, missing values handled, outliers capped, and saved for modeling.")
