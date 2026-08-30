# 🗳️ Portugal Election Prediction

A Streamlit web app that predicts `FinalMandates` (final number of seats won by a party in a territory) for the 2019 Portuguese legislative election, using a K-Nearest Neighbors model boosted with AdaBoost.


The model was trained in a separate notebook on the [Real-Time Election Results Portugal 2019](https://archive.ics.uci.edu/dataset/513/real+time+election+results+portugal+2019) dataset (UCI ML Repository). This repo contains only the deployment app and the exported model artifacts — not the training notebook.

---

## Model

Several regression models were evaluated during development (KNN, Linear Regression, Decision Tree, SVR, and a TensorFlow ANN). The **KNN Regressor boosted with AdaBoost** was selected as the final model for this app, based on the following test-set results:

| Model                      | MSE     | MAE    | R² Score  |
|:----------------------------|:--------|:-------|:----------|
| **KNN Regressor (AdaBoost)**| 0.0002  | 0.0009 | 0.999996  |

It was chosen over the alternatives (Linear Regression, Decision Tree/Random Forest, SVR, ANN) for having the lowest error and highest R² across cross-validation.

> Predictions are continuous (regression) values — e.g. `1.22` — and are rounded to the nearest whole seat in the app, since `FinalMandates` is always an integer in reality.

---

## Preprocessing

The model expects input that has gone through the same preprocessing used during training:

1. **Feature scaling** — all numerical features are standardized with a fitted `StandardScaler` (mean=0, std=1).
2. **Categorical encoding** — `territoryName` and `Party` are converted with a fitted `OneHotEncoder(sparse_output=False)`.

These preprocessors were fit once during training and exported alongside the model, so the app applies `.transform()` (never `.fit()`) to keep inference consistent with training. The app reads the valid `territoryName` / `Party` category lists directly from the fitted encoder, so the dropdown options can never drift out of sync with what the model was trained on.

Features removed before training (not used by the app either): `Hondt`, `Mandates`, `TimeElapsed`, `blankVotesPercentage`, `pre.blankVotesPercentage`, `nullVotesPercentage`, `pre.nullVotesPercentage`, `time`.

---

## Project Structure

```
portugal-election-prediction/
│
├── app.py                     # Streamlit app
├── model/
│   ├── knn_boost_model.pkl    # Trained AdaBoost + KNN regressor
│   ├── scaler.pkl             # Fitted StandardScaler
│   └── encoder.pkl            # Fitted OneHotEncoder
├── database/
│   └── users.db
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd portugal-election-prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub, including `model/knn_boost_model.pkl`, `model/scaler.pkl`, and `model/encoder.pkl` (use Git LFS if any file exceeds GitHub's 100MB limit).
2. Make sure `requirements.txt` pins the exact package versions used during training — especially `scikit-learn`, since a version mismatch between training and deployment can break pickle loading.
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click **New app**, select this repository/branch, and set the main file path to `app.py`.
5. Click **Deploy** and check the build logs if anything fails (missing dependency or wrong model path are the most common issues).
6. Once live, test a few predictions to confirm the model/scaler/encoder loaded correctly.
7. Copy the generated `https://<your-app-name>.streamlit.app` URL and update the **Live app** link at the top of this README.

---

## How Prediction Works

1. The user fills in election details (territory, party, mandates, parish, current/previous voting figures) through the form.
2. `app.py` loads three artifacts from `model/`: the fitted `scaler.pkl`, `encoder.pkl`, and `knn_boost_model.pkl`.
3. Numeric inputs are scaled with `scaler.transform(...)` and categorical inputs (`territoryName`, `Party`) are encoded with `encoder.transform(...)`, mirroring the exact preprocessing done in training.
4. The processed input is passed to the model for prediction.
5. The raw regression output is rounded to the nearest whole number and displayed as the predicted final seat count.

---

## Notes

- `model/scaler.pkl`, `model/encoder.pkl`, and `model/knn_boost_model.pkl` must all come from the **same training run** — mixing artifacts from different runs will produce incorrect predictions.
- Dataset source: [UCI Machine Learning Repository — Real-Time Election Results Portugal 2019](https://archive.ics.uci.edu/dataset/513/real+time+election+results+portugal+2019).
