from huggingface_hub import hf_hub_download

import streamlit as st
import torch

from huggingface_hub import hf_hub_download

from model import GAT
from explain import (
    create_explainer,
    explain_transaction
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GAT Fraud Detection",
    page_icon="🔍",
    layout="wide"
)


st.title("🔍 Financial Fraud Detection using GAT")

st.write(
    "Enter a transaction ID to obtain a fraud prediction "
    "and explainability results."
)


# --------------------------------------------------
# LOAD GRAPH DATA
# --------------------------------------------------

@st.cache_resource
def load_data():

    data_path = hf_hub_download(
        repo_id="PujithaDumpa/elliptic-gat-data",
        filename="elliptic_data.pt",
        repo_type="model"
    )

    data = torch.load(
        data_path,
        weights_only=False
    )

    return data


data = load_data()


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(data):

    model = GAT(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=2
    )

    model.load_state_dict(
        torch.load(
            "best_gat_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model(data)


# --------------------------------------------------
# CREATE EXPLAINER
# --------------------------------------------------

@st.cache_resource
def load_explainer(model):

    return create_explainer(model)


explainer = load_explainer(model)


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

node_id = st.number_input(
    "Enter Transaction ID",
    min_value=0,
    max_value=data.num_nodes - 1,
    value=1234,
    step=1
)


# --------------------------------------------------
# EXPLAIN BUTTON
# --------------------------------------------------

if st.button(
    "Explain Transaction",
    type="primary"
):

    with st.spinner(
        "Generating prediction and explanation..."
    ):

        result = explain_transaction(
            model,
            data,
            explainer,
            int(node_id)
        )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    prediction = result["prediction"]

    confidence = result["confidence"]


    if prediction == 0:

        st.error("🚨 Fraudulent Transaction")

    else:

        st.success("✅ Legitimate Transaction")


    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )


    # --------------------------------------------------
    # IMPORTANT NEIGHBORS
    # --------------------------------------------------

    st.subheader(
        "Top Influential Neighbors"
    )


    for neighbor, attention in result["neighbors"]:

        st.write(
            f"Transaction **{neighbor}** — "
            f"Attention: **{attention:.4f}**"
        )


    # --------------------------------------------------
    # IMPORTANT FEATURES
    # --------------------------------------------------

    st.subheader(
        "Top Important Features"
    )


    for feature, importance in result["features"]:

        st.write(
            f"Feature **{feature}** — "
            f"Importance: **{importance:.4f}**"
        )
