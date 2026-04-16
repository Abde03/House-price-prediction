import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    model_path = Path("models/best_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    else:
        st.error("Model not found. Please train the model first.")
        return None


@st.cache_data
def load_training_data():
    """Load cleaned training data for reference."""
    data_path = Path("data/processed/cleaned_data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Encode categorical columns (match training)
        from preprocess import encode_categorical_columns
        df_selected = df[[
            'price', 'rooms', 'bathrooms', 'area', 'elevator',
            'floor', 'terrace', 'parking', 'type', 'city', 'neighborhood'
        ]]
        df_encoded = encode_categorical_columns(df_selected)
        return df, df_encoded
    return None, None


@st.cache_data
def load_feature_schema():
    """Load feature order generated during training."""
    schema_path = Path("models/feature_schema.json")
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("feature_columns", [])
    return []


@st.cache_data
def load_model_metadata():
    """Load the metadata written by the training pipeline."""
    metadata_path = Path("models/model_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_feature_distributions(df_encoded):
    """Extract feature ranges for validation."""
    numeric_cols = ['rooms', 'bathrooms', 'area', 'floor']
    distributions = {}
    for col in numeric_cols:
        if col in df_encoded.columns:
            distributions[col] = {
                'min': df_encoded[col].min(),
                'max': df_encoded[col].max(),
                'mean': df_encoded[col].mean()
            }
    return distributions


def main():
    st.title("🏠 Morocco House Price Predictor")
    st.markdown("---")

    # Load model
    model = load_model()
    if model is None:
        return

    df_raw, df_encoded = load_training_data()
    distributions = get_feature_distributions(df_encoded) if df_encoded is not None else {}
    feature_columns = load_feature_schema()
    model_metadata = load_model_metadata()
    best_model_name = model_metadata.get("best_model_name", "Random Forest Regressor")
    best_cv_metrics = model_metadata.get("best_cv_metrics", {})

    if not feature_columns and df_encoded is not None:
        feature_columns = [c for c in df_encoded.columns if c != "price"]

    # Sidebar for inputs
    st.sidebar.header("📋 House Features")
    st.sidebar.markdown("Enter the property details below:")

    # Collect user inputs
    rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
    area = st.sidebar.number_input(
        "Land Area (m²)",
        min_value=10,
        max_value=5000,
        value=150
    )
    floor = st.sidebar.number_input("Floor Level", min_value=0, max_value=30, value=3)

    elevator = st.sidebar.selectbox("Has Elevator? (1=Yes, 0=No)", [0, 1])
    terrace = st.sidebar.selectbox("Has Terrace? (1=Yes, 0=No)", [0, 1])
    parking = st.sidebar.selectbox("Has Parking? (1=Yes, 0=No)", [0, 1])

    property_type = st.sidebar.selectbox(
        "Property Type",
        ["appartement", "studio", "villa", "bureau"],
        index=0
    )

    city = st.sidebar.selectbox(
        "City",
        sorted(df_raw["city"].dropna().unique().tolist()) if df_raw is not None else ["casablanca"],
        index=0
    )

    neighborhood = st.sidebar.selectbox(
        "Neighborhood",
        sorted(df_raw["neighborhood"].dropna().unique().tolist()) if df_raw is not None else ["autre"],
        index=0
    )

    st.sidebar.markdown("---")

    # Create input array with proper encoding
    input_dict = {
        'rooms': rooms,
        'bathrooms': bathrooms,
        'area': area,
        'elevator': elevator,
        'floor': floor,
        'terrace': terrace,
        'parking': parking,
    }

    # Build dynamic one-hot inputs from trained feature schema.
    for col in feature_columns:
        if col.startswith('city_'):
            city_name = col.replace('city_', '')
            input_dict[col] = 1 if city == city_name else 0
        elif col.startswith('neighborhood_'):
            neighborhood_name = col.replace('neighborhood_', '')
            input_dict[col] = 1 if neighborhood == neighborhood_name else 0
        elif col.startswith('type_'):
            type_name = col.replace('type_', '')
            input_dict[col] = 1 if property_type == type_name else 0
        elif col not in input_dict:
            input_dict[col] = 0

    # Create DataFrame with correct column order
    X_input = pd.DataFrame([input_dict])
    if feature_columns:
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Prediction Result")

        if st.button("🔍 Predict Price", use_container_width=True):
            prediction = model.predict(X_input)[0]

            # Display prediction
            st.success(f"### Estimated Price: **{prediction:,.2f} MAD**")

            # Price range estimate
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Estimated Price",
                    f"{prediction:,.0f} MAD",
                    delta=f"Based on {best_model_name}"
                )
            with col_b:
                # Show price per m2
                price_per_m2 = prediction / area
                st.metric(
                    "Price per m²",
                    f"{price_per_m2:,.0f} MAD/m²"
                )

            # Show feature summary
            st.subheader("📝 Input Summary")
            summary_df = pd.DataFrame(
                {
                    "Feature": [
                        "Rooms",
                        "Bathrooms",
                        "Area (m²)",
                        "Floor",
                        "Elevator",
                        "Terrace",
                        "Parking",
                        "Type",
                        "City",
                        "Neighborhood"
                    ],
                    "Value": [
                        rooms,
                        bathrooms,
                        area,
                        floor,
                        "Yes" if elevator == 1 else "No",
                        "Yes" if terrace == 1 else "No",
                        "Yes" if parking == 1 else "No",
                        property_type.capitalize(),
                        city.capitalize(),
                        neighborhood.capitalize()
                    ]
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("ℹ️ Model Info")
        st.info(
            """
**Best model:** {best_model_name}

**Inference features:** Dynamic from trained schema

**Validation:** Group-aware CV + holdout

**Tip:** retrain after changing preprocessing to refresh schema
""".format(best_model_name=best_model_name)
        )

        if best_cv_metrics:
            st.metric("Best CV R²", f"{best_cv_metrics.get('cv_r2_mean', 0.0):.4f}")
            st.caption(
                f"CV R² std: {best_cv_metrics.get('cv_r2_std', 0.0):.4f}"
            )

    st.markdown("---")

    # Statistics section
    if df_raw is not None:
        st.subheader("📈 Training Data Statistics")

        # Show dataset overview
        tab1, tab2, tab3 = st.tabs(["Dataset Info", "Price Distribution", "Feature Ranges"])

        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples (After Dedup)", len(df_encoded))
            with col2:
                st.metric("Min Price", f"{df_raw['price'].min():,.0f} MAD")
            with col3:
                st.metric("Max Price", f"{df_raw['price'].max():,.0f} MAD")
            with col4:
                st.metric("Avg Price", f"{df_raw['price'].mean():,.0f} MAD")

        with tab2:
            # Price distribution plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df_raw['price'], bins=30, edgecolor='k', alpha=0.7, color='steelblue')
            ax.axvline(df_raw['price'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.set_xlabel('Price (MAD)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Training Data: Price Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            # Feature ranges
            range_data = []
            for feature, stats in distributions.items():
                range_data.append({
                    "Feature": feature.capitalize(),
                    "Min": f"{stats['min']:.2f}",
                    "Max": f"{stats['max']:.2f}",
                    "Mean": f"{stats['mean']:.2f}"
                })
            if range_data:
                st.dataframe(pd.DataFrame(range_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 12px;'>"
        "🏠 Morocco House Price Predictor | Powered by MLflow & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
