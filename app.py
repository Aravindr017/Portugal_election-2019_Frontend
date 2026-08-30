import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Portugal Election Prediction",
    page_icon="🗳️",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "knn_boost_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"


# =========================================================
# LOAD MODEL + PREPROCESSORS
#
# These are three SEPARATE pickles, all produced from the
# same notebook run: the fitted StandardScaler, the fitted
# OneHotEncoder, and the fitted regressor. All three must
# come from the same training run or the columns/scaling
# won't line up with what the model expects.
# =========================================================

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


model, scaler, encoder = load_artifacts()

# Columns each preprocessor was fit on (sklearn stores this automatically
# when fit on a DataFrame) — avoids hard-coding the lists a second time.
numeric_columns = list(scaler.feature_names_in_)
cat_cols = list(encoder.feature_names_in_)


# =========================================================
# RAW FEATURE NAMES
# These are the raw input columns, BEFORE scaling/encoding.
# =========================================================

feature_names = [
    "territoryName",
    "Party",
    "totalMandates",
    "availableMandates",
    "numParishes",
    "numParishesApproved",
    "blankVotes",
    "nullVotes",
    "votersPercentage",
    "subscribedVoters",
    "totalVoters",
    "pre.blankVotes",
    "pre.nullVotes",
    "pre.votersPercentage",
    "pre.subscribedVoters",
    "pre.totalVoters",
    "Percentage",
    "validVotesPercentage",
    "Votes"
]


# =========================================================
# PULL VALID CATEGORIES DIRECTLY FROM THE FITTED ENCODER
#
# This is the key fix for the original bug: instead of
# hard-coding territory/party lists that can drift out of
# sync with what the model was actually trained on, we read
# the categories straight off the fitted OneHotEncoder.
# They are guaranteed to match (right spelling, right order,
# right punctuation e.g. "B.E." vs "BE").
# =========================================================

category_options = dict(zip(cat_cols, [list(c) for c in encoder.categories_]))
territory_options = category_options.get("territoryName", [])
party_options = category_options.get("Party", [])


# =========================================================
# NUMERIC DEFAULTS
# =========================================================

numeric_defaults = {
    "totalMandates": 10.0,
    "availableMandates": 10.0,
    "numParishes": 100.0,
    "numParishesApproved": 100.0,
    "blankVotes": 0.0,
    "nullVotes": 0.0,
    "votersPercentage": 50.0,
    "subscribedVoters": 10000.0,
    "totalVoters": 5000.0,
    "pre.blankVotes": 0.0,
    "pre.nullVotes": 0.0,
    "pre.votersPercentage": 50.0,
    "pre.subscribedVoters": 10000.0,
    "pre.totalVoters": 5000.0,
    "Percentage": 10.0,
    "validVotesPercentage": 10.0,
    "Votes": 500.0
}


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title("🗳️ Portugal Election Prediction")
st.caption("Machine Learning Powered Election Prediction System")
st.divider()


# =========================================================
# USER INPUT
# =========================================================

st.subheader("Enter Election Information")

input_data = {}


# =========================================================
# ELECTION INFORMATION
# =========================================================

with st.expander("🏛️ Election Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        input_data["territoryName"] = st.selectbox(
            "Territory Name",
            options=territory_options
        )

    with col2:
        input_data["Party"] = st.selectbox(
            "Political Party",
            options=party_options
        )

    col1, col2 = st.columns(2)

    with col1:
        input_data["totalMandates"] = st.number_input(
            "Total Mandates",
            value=numeric_defaults["totalMandates"],
            min_value=0.0
        )

    with col2:
        input_data["availableMandates"] = st.number_input(
            "Available Mandates",
            value=numeric_defaults["availableMandates"],
            min_value=0.0
        )


# =========================================================
# PARISH INFORMATION
# =========================================================

with st.expander("🏘️ Parish Information"):

    col1, col2 = st.columns(2)

    with col1:
        input_data["numParishes"] = st.number_input(
            "Number of Parishes",
            value=numeric_defaults["numParishes"],
            min_value=0.0
        )

    with col2:
        input_data["numParishesApproved"] = st.number_input(
            "Approved Parishes",
            value=numeric_defaults["numParishesApproved"],
            min_value=0.0
        )


# =========================================================
# CURRENT VOTING INFORMATION
# =========================================================

with st.expander("🗳️ Current Voting Information"):

    col1, col2 = st.columns(2)

    current_features = [
        "blankVotes",
        "nullVotes",
        "votersPercentage",
        "subscribedVoters",
        "totalVoters"
    ]

    for index, feature in enumerate(current_features):
        column = col1 if index % 2 == 0 else col2
        with column:
            input_data[feature] = st.number_input(
                feature,
                value=numeric_defaults[feature],
                min_value=0.0
            )


# =========================================================
# PREVIOUS ELECTION INFORMATION
# =========================================================

with st.expander("📊 Previous Election Information"):

    col1, col2 = st.columns(2)

    previous_features = [
        "pre.blankVotes",
        "pre.nullVotes",
        "pre.votersPercentage",
        "pre.subscribedVoters",
        "pre.totalVoters"
    ]

    for index, feature in enumerate(previous_features):
        column = col1 if index % 2 == 0 else col2
        with column:
            input_data[feature] = st.number_input(
                feature,
                value=numeric_defaults[feature],
                min_value=0.0
            )


# =========================================================
# PARTY VOTING INFORMATION
# =========================================================

with st.expander("📈 Party Voting Information"):

    col1, col2 = st.columns(2)

    party_features = [
        "Percentage",
        "validVotesPercentage",
        "Votes"
    ]

    for index, feature in enumerate(party_features):
        column = col1 if index % 2 == 0 else col2
        with column:
            input_data[feature] = st.number_input(
                feature,
                value=numeric_defaults[feature],
                min_value=0.0
            )


# =========================================================
# PREDICTION
# =========================================================

st.divider()

if st.button("🔮 Predict Final Mandates", use_container_width=True):

    try:
        # ---------------------------------------------
        # 1. Build the RAW input dataframe from user input
        # ---------------------------------------------
        raw_input = pd.DataFrame([input_data])[feature_names]

        with st.expander("🔍 View Raw Input Data"):
            st.dataframe(raw_input, use_container_width=True)

        # ---------------------------------------------
        # 2. Scale numeric columns using the SAME fitted
        #    scaler from training (transform, not fit_transform)
        # ---------------------------------------------
        processed_input = raw_input.copy()
        processed_input[numeric_columns] = scaler.transform(
            raw_input[numeric_columns]
        )

        # ---------------------------------------------
        # 3. One-hot encode categorical columns using the
        #    SAME fitted encoder from training
        # ---------------------------------------------
        encoded_array = encoder.transform(raw_input[cat_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(cat_cols),
            index=raw_input.index
        )

        # ---------------------------------------------
        # 4. Combine, dropping the original raw categorical
        #    columns — mirrors exactly what the notebook did:
        #    X = X.drop(columns=cat_cols); X = pd.concat([X, encoded_df])
        # ---------------------------------------------
        processed_input = processed_input.drop(columns=cat_cols)
        processed_input = pd.concat([processed_input, encoded_df], axis=1)

        with st.expander("⚙️ View Preprocessed Model Input"):
            st.dataframe(processed_input, use_container_width=True)

        # ---------------------------------------------
        # 5. Predict
        # ---------------------------------------------
        prediction = model.predict(processed_input)
        predicted_mandates = prediction[0]

        st.success("Prediction completed successfully!")

        rounded_mandates = max(0, round(predicted_mandates))

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Predicted Final Mandates",
                value=f"{rounded_mandates}"
            )
            st.caption(f"Raw model output: {predicted_mandates:.2f}")

    except Exception as error:
        st.error(f"Prediction Error: {error}")