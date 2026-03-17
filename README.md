# Sustainable Construction Optimizer (Streamlit)

Machine Learning–based carbon emission optimization for construction project scenarios.

## Prerequisites

- Python 3.10+ installed
- Git installed

## Clone

```bash
git clone https://github.com/mid-nair/eco-planner-.git
cd eco-planner-
```

## Create a virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets (required)

This project expects the following CSV files inside a `dataset/` folder in the project root:

```
dataset/
  materials_dataset_1000.csv
  scenario_dataset_1000.csv
  location_dataset_1000.csv
  equipment_dataset_1000.csv
  construction_schedule_dataset_1000.csv
  construction_material_carbon_dataset_india (1) (1).csv
```

Notes:
- The repository ignores `dataset/` and `*.csv` by default (to avoid GitHub file-size limits). Copy these files manually onto the machine after cloning.
- Paths are configured in `app/config.py` under `class Paths`.

## Run the app

```bash
python -m streamlit run app/main_app.py
```

If `streamlit` is not found on Windows, make sure your venv is activated, or use:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main_app.py
```

## What you’ll see

- Train a single model (choose algorithm in the sidebar)
- Compare all algorithms on the same train/test split (Step 1 → **Compare algorithms**)
- Predict emissions for scenarios, optimize plans, and view TOPSIS-ranked results

