# AI/ML Starter Project

[![Tests](https://github.com/SwaminathanV510/ai-ml-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/SwaminathanV510/ai-ml-starter/actions)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A beginner-friendly machine learning project using Python and scikit-learn. It trains a classifier on the classic Iris dataset, saves the model, includes tests, and provides a small FastAPI service for predictions.

## Quickstart

### 1) Clone & set up
```bash
git clone <your-repo-url>
cd ai-ml-starter
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train a model
```bash
python -m src.train
```
This will train a RandomForest classifier and save it to `models/model.joblib`, along with a printed accuracy.

### 3) Run tests (optional but recommended)
```bash
pytest -q
```

### 4) Launch the prediction API
```bash
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs` for an interactive Swagger UI.

### 5) Make a prediction (example)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Repo structure

```
ai-ml-starter/
├─ app/
│  └─ main.py           # FastAPI app for serving predictions
├─ models/              # Saved models (ignored by git)
├─ notebooks/
│  └─ EDA.ipynb         # Starter notebook
├─ src/
│  ├─ __init__.py
│  ├─ data.py           # Data loading & split
│  ├─ train.py          # Train & save model
│  └─ predict.py        # Load model & predict from CLI
├─ tests/
│  ├─ test_data.py
│  └─ test_train.py
├─ .github/workflows/ci.yml
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## How to create the GitHub repo

1. Create a new empty repo on GitHub (no README, no .gitignore).
2. Locally:
```bash
cd ai-ml-starter
git init
git add .
git commit -m "Initial commit: AI/ML starter"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Next steps & ideas
- Swap Iris for your own dataset (put CSVs under `data/` and adjust `src/data.py`).
- Try different models (LogisticRegression, XGBoost, etc.).
- Track experiments (MLflow), add notebooks, and enable pre-commit hooks.
- Deploy the FastAPI app (Railway, Render, Fly.io, etc.).
