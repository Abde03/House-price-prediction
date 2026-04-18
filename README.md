# Morocco House Pricing

End-to-end machine learning project to predict house prices in Morocco using cleaned tabular data, model comparison, MLflow tracking, and a Streamlit app.

## Project Structure

- `src/preprocess.py`: Data cleaning and feature preparation.
- `src/train.py`: Model training, evaluation, and model comparison artifacts.
- `src/pipeline.py`: Runs preprocessing + training in one command.
- `src/app.py`: Streamlit prediction dashboard.
- `data/raw/houses_data_eng.csv`: Input raw dataset.
- `data/processed/cleaned_data.csv`: Cleaned dataset used for training.
- `models/`: Saved model and metadata.
- `results/`: Generated plots and model comparison results.
- `mlruns/`: MLflow experiment tracking artifacts.

## Prerequisites

- Python 3.11+
- Git (optional)
- PowerShell (Windows)

## Step-by-Step: Run the Project (Windows)

1. Open a terminal in the project root:

```powershell
cd C:\Users\Azzao\OneDrive\Bureau\house_pricing
```

2. Create a virtual environment:

```powershell
python -m venv .venv
```

3. Activate the virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the full data + training pipeline:

```powershell
python src/pipeline.py
```

This generates:

- `models/best_model.pkl`
- `models/feature_schema.json`
- `models/model_metadata.json`
- `results/model_comparison.csv`
- `results/model_comparison.png`
- model-specific plots in `results/` (feature importance, residuals, actual vs predicted)

6. Start the Streamlit app:

```powershell
streamlit run src/app.py
```

7. Open your browser at:

- `http://localhost:8501`

## Run Individual Steps (Optional)

### Preprocess only

```powershell
python src/preprocess.py
```

### Train only

```powershell
python src/train.py
```

## MLflow Tracking (Optional)

Start the MLflow UI from project root:

```powershell
mlflow ui
```

Then open:

- `http://localhost:5000`

## Docker (Optional)

1. Build image:

```powershell
docker build -t house-pricing-app .
```

2. Run container:

```powershell
docker run --rm -p 8501:8501 house-pricing-app
```

3. Open:

- `http://localhost:8501`

## Notes

- If the raw dataset is missing, `src/pipeline.py` reuses `data/processed/cleaned_data.csv` if available.
- Model comparison in the app is shown under the Visual Insights section.
