import streamlit as st
import torch
from huggingface_hub import hf_hub_download

from model import GAT
from explain import create_explainer, explain_transaction


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
# DEVICE
# --------------------------------------------------

DEVICE = torch.device("cpu")


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
        map_location=DEVICE,
        weights_only=False
    )

    data = data.to(DEVICE)

    return data


with st.spinner("Loading graph data..."):
    data = load_data()


st.success(
    f"Graph loaded: {data.num_nodes:,} transactions"
)


# --------------------------------------------------
# LOAD TRAINED GAT MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(num_features):

    model = GAT(
        in_channels=num_features,
        hidden_channels=128,
        out_channels=2
    )

    state_dict = torch.load(
        "best_gat_model.pth",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


with st.spinner("Loading trained GAT model..."):
    model = load_model(data.num_features)


st.success("GAT model loaded successfully.")


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

node_id = st.number_input(
    "Enter Transaction ID",
    min_value=0,
    max_value=int(data.num_nodes - 1),
    value=0,
    step=1
)


# --------------------------------------------------
# EXPLAIN BUTTON
# --------------------------------------------------

if st.button(
    "Explain Transaction",
    type="primary"
):

    # --------------------------------------------------
    # CREATE EXPLAINER ONLY WHEN NEEDED
    # --------------------------------------------------

    with st.spinner("Preparing GNN explainer..."):

        explainer = create_explainer(model)


    # --------------------------------------------------
    # GENERATE EXPLANATION
    # --------------------------------------------------

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

        st.error(
            "🚨 Fraudulent Transaction"
        )

    else:

        st.success(
            "✅ Legitimate Transaction"
        )


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


    if result["neighbors"]:

        for neighbor, attention in result["neighbors"]:

            st.write(
                f"Transaction **{neighbor}** — "
                f"Attention: **{attention:.4f}**"
            )

    else:

        st.write(
            "No influential neighbors found."
        )


    # --------------------------------------------------
    # IMPORTANT FEATURES
    # --------------------------------------------------

    st.subheader(
        "Top Important Features"
    )


    if result["features"]:

        for feature, importance in result["features"]:

            st.write(
                f"Feature **{feature}** — "
                f"Importance: **{importance:.4f}**"
            )

    else:

        st.write(
            "No important features found."
        )
