import pandas as pd
from pathlib import Path


COLUMN_MAPPING = {
    "new_price": "price",
    "chambres": "rooms",
    "salles de bains": "bathrooms",
    "surface": "area",
    "ascenseur": "elevator",
    "terrasse": "terrace",
    "parking": "parking",
    "Type": "type",
    "City": "city",
    "Nighberd": "neighborhood",
}


YES_NO_MAPPING = {
    "Yes": 1,
    "No": 0,
    "yes": 1,
    "no": 0,
}


REQUIRED_COLUMNS = [
    "price",
    "rooms",
    "bathrooms",
    "area",
    "elevator",
    "floor",
    "terrace",
    "parking",
    "type",
    "city",
    "neighborhood",
]


def load_data(file_path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    return pd.read_csv(file_path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns into clean English names."""
    return df.rename(columns=COLUMN_MAPPING)


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove useless index-like or text columns not needed for training."""
    columns_to_drop = ["Unnamed: 0.1", "Unnamed: 0", "desc", "address"]
    return df.drop(columns=columns_to_drop, errors="ignore")


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text columns for consistency."""
    for col in ["city", "type", "neighborhood"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df


def map_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yes/No columns into 1/0."""
    for col in ["elevator", "terrace", "parking"]:
        if col in df.columns:
            df[col] = df[col].map(YES_NO_MAPPING)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns safely."""
    numeric_columns = ["price", "rooms", "bathrooms", "area", "floor"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def remove_missing_and_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or invalid values."""
    df = df.dropna(subset=["price", "rooms", "bathrooms", "area", "floor", "city", "type"])

    df = df[
        (df["price"] > 0) &
        (df["area"] > 0) &
        (df["rooms"] >= 0) &
        (df["bathrooms"] >= 0)
    ]
    return df


def keep_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns useful for modeling."""
    available_columns = [col for col in REQUIRED_COLUMNS if col in df.columns]
    return df[available_columns].copy()


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    categorical_cols = ["city", "type", "neighborhood"]
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    return pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Complete preprocessing pipeline."""
    df = load_data(input_path)
    df = rename_columns(df)
    df = drop_useless_columns(df)
    df = normalize_text_columns(df)
    df = map_binary_columns(df)
    df = convert_numeric_columns(df)
    df = keep_required_columns(df)
    df = remove_missing_and_invalid_rows(df)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


def preprocess_and_encode(input_path: str, cleaned_output_path: str) -> pd.DataFrame:
    """Preprocess raw data, then encode categorical features."""
    df = preprocess_data(input_path, cleaned_output_path)
    df_encoded = encode_categorical_columns(df)
    return df_encoded


if __name__ == "__main__":
    raw_file = "data/raw/houses_data_eng.csv"
    cleaned_file = "data/processed/cleaned_data.csv"

    cleaned_df = preprocess_data(raw_file, cleaned_file)

    print("Preprocessing completed successfully.")
    print("Cleaned shape:", cleaned_df.shape)
    print(cleaned_df.head())