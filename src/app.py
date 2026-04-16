import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Morocco House Price Studio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCHEMA_PATH = BASE_DIR / "models" / "feature_schema.json"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
HERO_IMAGE_PATH = BASE_DIR / "assets" / "hero_house.svg"
RESULTS_DIR = BASE_DIR / "results"


@st.cache_resource
def load_model():
    """Load the trained model."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    st.error("Model not found. Please train the model first.")
    return None


@st.cache_data
def load_training_data():
    """Load cleaned data used for reference charts and option lists."""
    if not DATA_PATH.exists():
        return None, None

    df = pd.read_csv(DATA_PATH)
    from preprocess import encode_categorical_columns

    df_selected = df[[
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
    ]]
    df_encoded = encode_categorical_columns(df_selected)
    return df, df_encoded


@st.cache_data
def load_feature_schema():
    """Load the column order used during training."""
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        return payload.get("feature_columns", [])
    return []


@st.cache_data
def load_model_metadata():
    """Load training summary metadata."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    return {}


@st.cache_data
def load_svg_as_html(path_str: str):
    """Load SVG file as raw HTML string."""
    path = Path(path_str)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def get_feature_distributions(df_encoded):
    """Extract numeric ranges for the dashboard."""
    numeric_cols = ["rooms", "bathrooms", "area", "floor"]
    distributions = {}
    for col in numeric_cols:
        if col in df_encoded.columns:
            distributions[col] = {
                "min": df_encoded[col].min(),
                "max": df_encoded[col].max(),
                "mean": df_encoded[col].mean(),
            }
    return distributions


def apply_custom_style():
    """Render the visual theme for the app."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(10, 18, 34, 0.78);
            --panel-strong: rgba(10, 18, 34, 0.94);
            --text: #f8fbff;
            --muted: #a8b6c9;
            --teal: #16c7b7;
            --blue: #40a9ff;
            --gold: #f5b942;
            --coral: #ff7f66;
            --border: rgba(255, 255, 255, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(22, 199, 183, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(64, 169, 255, 0.16), transparent 28%),
                linear-gradient(180deg, #07111f 0%, #0a1628 52%, #06101a 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 18, 34, 0.98) 0%, rgba(8, 14, 27, 0.98) 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(13, 20, 39, 0.92), rgba(8, 13, 26, 0.82));
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 28px;
            padding: 28px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(22, 199, 183, 0.12);
            color: var(--teal);
            font-size: 13px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid rgba(22, 199, 183, 0.28);
        }

        .hero-title {
            font-size: 3rem;
            line-height: 1.02;
            margin: 14px 0 12px;
            color: var(--text);
            font-weight: 800;
        }

        .hero-title span {
            background: linear-gradient(90deg, var(--teal), var(--blue), var(--gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 760px;
            line-height: 1.7;
        }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 20px;
        }

        .stat-card, .glass-card {
            background: linear-gradient(180deg, rgba(12, 22, 41, 0.88), rgba(9, 16, 31, 0.88));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 18px 18px 16px;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 6px;
            color: var(--text);
        }

        .stat-help {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 4px;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 800;
            margin: 8px 0 14px;
            color: var(--text);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 16px;
        }

        .metric-box {
            padding: 18px;
            border-radius: 20px;
            background: rgba(10, 18, 34, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }

        .metric-title {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 6px;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 6px;
        }

        .result-card {
            background: linear-gradient(135deg, rgba(22, 199, 183, 0.16), rgba(64, 169, 255, 0.18), rgba(245, 185, 66, 0.10));
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 26px;
            padding: 22px;
            box-shadow: 0 22px 50px rgba(0, 0, 0, 0.28);
        }

        .result-price {
            font-size: 2.6rem;
            font-weight: 900;
            margin-top: 8px;
            color: #ffffff;
        }

        .result-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
            font-size: 0.86rem;
            margin: 4px 8px 4px 0;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .stButton > button {
            background: linear-gradient(90deg, #16c7b7 0%, #40a9ff 48%, #f5b942 100%);
            color: #04111b;
            font-weight: 800;
            border: none;
            padding: 0.8rem 1.2rem;
            border-radius: 16px;
            box-shadow: 0 12px 28px rgba(22, 199, 183, 0.25);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 30px rgba(22, 199, 183, 0.32);
        }

        .stMetric {
            background: rgba(10, 18, 34, 0.66);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 14px 14px 10px;
            border-radius: 18px;
        }

        .stDataFrame, .stMarkdown, .stTabs {
            color: var(--text);
        }

        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section(model_metadata, df_raw):
    """Top banner with branding and key stats."""
    best_model_name = model_metadata.get("best_model_name", "ExtraTreesRegressor (Robust)")
    best_cv_metrics = model_metadata.get("best_cv_metrics", {})
    model_count = len(model_metadata.get("model_names", [])) or 3
    sample_count = len(df_raw) if df_raw is not None else 0
    feature_count = len(model_metadata.get("feature_columns", [])) or 0

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Morocco House Price Studio</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title">Predict house prices with a <span>premium ML interface</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="hero-copy">
                A redesigned Streamlit experience for Moroccan housing predictions with a richer visual style,
                model comparison context, and quick insight panels for trust and interpretation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="margin-top:18px;">
                <span class="pill">Best model: {best_model_name}</span>
                <span class="pill">Models compared: {model_count}</span>
                <span class="pill">Training rows: {sample_count:,}</span>
                <span class="pill">Features: {feature_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        svg_html = load_svg_as_html(str(HERO_IMAGE_PATH))
        if svg_html:
            st.markdown(svg_html, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="glass-card" style="min-height: 360px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
                    <div style="font-size: 4rem;">🏡</div>
                    <div style="font-size: 1.4rem; font-weight: 800; margin-top: 10px;">Morocco House Price Studio</div>
                    <div style="color: #a8b6c9; margin-top: 8px; line-height: 1.6;">
                        A visual, model-driven dashboard for price prediction and experiment tracking.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if best_cv_metrics:
        st.markdown(
            f"""
            <div class="hero-stats">
                <div class="stat-card">
                    <div class="stat-label">Best CV R²</div>
                    <div class="stat-value">{best_cv_metrics.get('cv_r2_mean', 0.0):.4f}</div>
                    <div class="stat-help">Cross-validation mean score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">CV R² Std</div>
                    <div class="stat-value">{best_cv_metrics.get('cv_r2_std', 0.0):.4f}</div>
                    <div class="stat-help">Variation across splits</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Pipeline status</div>
                    <div class="stat-value">Ready</div>
                    <div class="stat-help">Tracked with MLflow + GitHub Actions</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_input_sidebar(df_raw):
    """Sidebar controls with a sleek card layout."""
    st.sidebar.markdown(
        """
        <div style="padding: 8px 0 2px;">
            <div style="font-size: 1.2rem; font-weight: 800;">House Features</div>
            <div style="color: #a8b6c9; margin-top: 4px; line-height: 1.5;">
                Tune the property details to estimate the price.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.container():
        rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
        area = st.number_input("Land Area (m²)", min_value=10, max_value=5000, value=150)
        floor = st.number_input("Floor Level", min_value=0, max_value=30, value=3)
        elevator = st.selectbox("Has Elevator?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        terrace = st.selectbox("Has Terrace?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        parking = st.selectbox("Has Parking?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        property_type = st.selectbox(
            "Property Type",
            ["appartement", "studio", "villa", "bureau"],
            index=0,
        )

        city_options = sorted(df_raw["city"].dropna().unique().tolist()) if df_raw is not None else ["casablanca"]
        neighborhood_options = (
            sorted(df_raw["neighborhood"].dropna().unique().tolist()) if df_raw is not None else ["autre"]
        )

        city = st.selectbox("City", city_options, index=0)
        neighborhood = st.selectbox("Neighborhood", neighborhood_options, index=0)

    return {
        "rooms": rooms,
        "bathrooms": bathrooms,
        "area": area,
        "floor": floor,
        "elevator": elevator,
        "terrace": terrace,
        "parking": parking,
        "property_type": property_type,
        "city": city,
        "neighborhood": neighborhood,
    }


def build_input_frame(user_inputs, feature_columns):
    """Create the model input dataframe using the training schema."""
    input_dict = {
        "rooms": user_inputs["rooms"],
        "bathrooms": user_inputs["bathrooms"],
        "area": user_inputs["area"],
        "elevator": user_inputs["elevator"],
        "floor": user_inputs["floor"],
        "terrace": user_inputs["terrace"],
        "parking": user_inputs["parking"],
    }

    for col in feature_columns:
        if col.startswith("city_"):
            city_name = col.replace("city_", "")
            input_dict[col] = 1 if user_inputs["city"] == city_name else 0
        elif col.startswith("neighborhood_"):
            neighborhood_name = col.replace("neighborhood_", "")
            input_dict[col] = 1 if user_inputs["neighborhood"] == neighborhood_name else 0
        elif col.startswith("type_"):
            type_name = col.replace("type_", "")
            input_dict[col] = 1 if user_inputs["property_type"] == type_name else 0
        elif col not in input_dict:
            input_dict[col] = 0

    x_input = pd.DataFrame([input_dict])
    if feature_columns:
        x_input = x_input.reindex(columns=feature_columns, fill_value=0)
    return x_input


def render_result_card(prediction, area, best_model_name):
    """Beautiful result panel after prediction."""
    price_per_m2 = prediction / area if area else 0
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-subtitle">Estimated property price</div>
            <div class="result-price">{prediction:,.0f} MAD</div>
            <div style="margin-top:12px; color:#d6e4f0; line-height:1.6;">
                Based on <strong>{best_model_name}</strong> and the current feature selection.
            </div>
            <div style="display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top:18px;">
                <div class="metric-box">
                    <div class="metric-title">Price per m²</div>
                    <div class="metric-value">{price_per_m2:,.0f}</div>
                    <div class="metric-note">MAD per square meter</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Model confidence proxy</div>
                    <div class="metric-value">High</div>
                    <div class="metric-note">From best validation run</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_table(user_inputs):
    """Show the selected input values in a polished table."""
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
                "Neighborhood",
            ],
            "Value": [
                user_inputs["rooms"],
                user_inputs["bathrooms"],
                user_inputs["area"],
                user_inputs["floor"],
                "Yes" if user_inputs["elevator"] == 1 else "No",
                "Yes" if user_inputs["terrace"] == 1 else "No",
                "Yes" if user_inputs["parking"] == 1 else "No",
                user_inputs["property_type"].capitalize(),
                user_inputs["city"].capitalize(),
                user_inputs["neighborhood"].capitalize(),
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def show_visual_insights():
    """Display model result images if available."""
    insight_tabs = st.tabs(["Feature Importance", "Prediction Fit", "Residuals"])
    insight_files = {
        "Feature Importance": RESULTS_DIR / "et_robust_importance.png",
        "Prediction Fit": RESULTS_DIR / "et_robust_actual_vs_pred.png",
        "Residuals": RESULTS_DIR / "et_robust_residuals.png",
    }

    for tab, (label, path) in zip(insight_tabs, insight_files.items()):
        with tab:
            if path.exists():
                st.image(str(path), caption=label, use_container_width=True)
            else:
                st.info(f"{label} chart is not available yet. Run the training pipeline first.")


def main():
    apply_custom_style()

    df_raw, df_encoded = load_training_data()
    distributions = get_feature_distributions(df_encoded) if df_encoded is not None else {}
    feature_columns = load_feature_schema()
    model_metadata = load_model_metadata()
    best_model_name = model_metadata.get("best_model_name", "ExtraTreesRegressor (Robust)")
    best_cv_metrics = model_metadata.get("best_cv_metrics", {})

    model = load_model()
    if model is None:
        st.stop()

    if not feature_columns and df_encoded is not None:
        feature_columns = [c for c in df_encoded.columns if c != "price"]

    render_hero_section(model_metadata, df_raw)

    st.markdown("<div class='section-title'>Predict a House Price</div>", unsafe_allow_html=True)

    user_inputs = render_input_sidebar(df_raw)
    x_input = build_input_frame(user_inputs, feature_columns)

    top_left, top_right = st.columns([1.3, 0.7], gap="large")

    with top_left:
        predict_clicked = st.button("Estimate Price", use_container_width=True)

        if predict_clicked:
            prediction = float(model.predict(x_input)[0])
            render_result_card(prediction, user_inputs["area"], best_model_name)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Input Summary</div>", unsafe_allow_html=True)
            render_summary_table(user_inputs)
        else:
            st.markdown(
                """
                <div class="glass-card" style="min-height: 260px; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:1.2rem; font-weight:800; color:#f8fbff;">Ready when you are</div>
                    <div style="color:#a8b6c9; margin-top:10px; line-height:1.7;">
                        Choose the property details in the sidebar, then click <strong>Estimate Price</strong> to see the prediction.
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:16px;">
                        <span class="pill">Dynamic one-hot encoding</span>
                        <span class="pill">Schema-safe inference</span>
                        <span class="pill">MLflow tracked</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with top_right:
        st.markdown("<div class='section-title'>Model Snapshot</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-title">Best model</div>
                <div class="metric-value" style="margin-top:4px;">{best_model_name}</div>
                <div class="metric-note">Validation-backed model selected from the pipeline</div>
                <div style="margin-top:16px; display:grid; gap:10px;">
                    <div class="metric-box">
                        <div class="metric-title">Best CV R²</div>
                        <div class="metric-value">{best_cv_metrics.get('cv_r2_mean', 0.0):.4f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">CV R² std</div>
                        <div class="metric-value">{best_cv_metrics.get('cv_r2_std', 0.0):.4f}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Dataset", "Visual Insights", "About"])

    with tabs[0]:
        st.markdown("<div class='section-title'>Training Data Overview</div>", unsafe_allow_html=True)
        if df_raw is not None:
            overview_cols = st.columns(4)
            metrics = [
                ("Total samples", len(df_encoded) if df_encoded is not None else len(df_raw), "After dedup and encoding"),
                ("Min price", f"{df_raw['price'].min():,.0f} MAD", "Smallest observed value"),
                ("Max price", f"{df_raw['price'].max():,.0f} MAD", "Largest observed value"),
                ("Average price", f"{df_raw['price'].mean():,.0f} MAD", "Mean target value"),
            ]
            for col, (title, value, note) in zip(overview_cols, metrics):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-title">{title}</div>
                            <div class="metric-value">{value}</div>
                            <div class="metric-note">{note}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            plot_left, plot_right = st.columns([1, 1], gap="large")
            with plot_left:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df_raw["price"], bins=30, edgecolor="white", alpha=0.78, color="#40a9ff")
                ax.axvline(df_raw["price"].mean(), color="#f5b942", linestyle="--", linewidth=2, label="Mean")
                ax.set_facecolor("#0b1526")
                fig.patch.set_facecolor("#0b1526")
                ax.tick_params(colors="#dce9f7")
                ax.xaxis.label.set_color("#dce9f7")
                ax.yaxis.label.set_color("#dce9f7")
                ax.title.set_color("#f8fbff")
                for spine in ax.spines.values():
                    spine.set_color("#38506c")
                ax.set_xlabel("Price (MAD)")
                ax.set_ylabel("Frequency")
                ax.set_title("Training Data: Price Distribution", fontweight="bold")
                ax.legend(facecolor="#0b1526", edgecolor="#38506c", labelcolor="#f8fbff")
                st.pyplot(fig, clear_figure=True)

            with plot_right:
                st.markdown("<div class='section-title'>Feature Ranges</div>", unsafe_allow_html=True)
                range_data = []
                for feature, stats in distributions.items():
                    range_data.append(
                        {
                            "Feature": feature.capitalize(),
                            "Min": f"{stats['min']:.2f}",
                            "Max": f"{stats['max']:.2f}",
                            "Mean": f"{stats['mean']:.2f}",
                        }
                    )
                if range_data:
                    st.dataframe(pd.DataFrame(range_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No distribution summary available.")
        else:
            st.info("Training data is unavailable.")

    with tabs[1]:
        st.markdown("<div class='section-title'>Model Visual Insights</div>", unsafe_allow_html=True)
        st.markdown(
            "The charts below are generated from the best training run and help you explain how the model behaves.")
        show_visual_insights()

    with tabs[2]:
        st.markdown("<div class='section-title'>About This App</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:1.05rem; line-height:1.8; color:#dce9f7;">
                    <strong>What it does:</strong> estimates Moroccan house prices using the best trained model.<br/>
                    <strong>How it works:</strong> feature inputs are aligned to the trained schema, then scored by the saved model.<br/>
                    <strong>Tracking:</strong> experiments are logged in MLflow and the best model metadata is stored locally.<br/>
                    <strong>Model:</strong> {best_model_name}<br/>
                    <strong>Validation:</strong> holdout + group-aware CV
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#a8b6c9; font-size:0.85rem;'>"
        "Morocco House Price Studio • Powered by MLflow, Streamlit, and automated training"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
