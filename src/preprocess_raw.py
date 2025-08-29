import pandas as pd

RAW_DATA_PATH = "sofia-real-estate-mlops/datasets/raw_data.csv"
CLEAN_DATA_PATH = "sofia-real-estate-mlops/datasets/clean_data.csv"

# Load raw dataset
raw_df = pd.read_csv(RAW_DATA_PATH)

# Make a copy to work on
df = raw_df.copy()

# --- Cleaning ---

# 1. Location: keep only text before comma
df['location'] = df['location'].str.split(',').str[0].str.strip()

# 2. Price: keep only numbers, remove spaces and €
df['price'] = df['price'].str.replace(r'[^0-9]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce').astype('Int64')

# 3. Property type: keep only first word
df['property_type'] = df['property_type'].str.split().str[0]

# 4. Size: keep only numbers
df['size'] = df['size'].str.extract(r'(\d+)')
df['size'] = pd.to_numeric(df['size'], errors='coerce').astype('Int64')

# 5. Year built: keep only numbers
df['year_built'] = df['year_built'].str.extract(r'(\d{4})')
df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce').astype('Int64')

# 6. Floor number: keep only numbers (including negatives), map Партер → 0
df['floor_number'] = df['floor_number'].replace("Партер", "0")
df['floor_number'] = df['floor_number'].str.extract(r'(-?\d+)')
df['floor_number'] = pd.to_numeric(df['floor_number'], errors='coerce').astype('Int64')

# --- Save cleaned dataset ---
df.to_csv(CLEAN_DATA_PATH, index=False)

print(f"✅ Cleaning complete. Saved to {CLEAN_DATA_PATH}")
print(df.dtypes)