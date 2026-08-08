
import streamlit as st
import torch

from huggingface_hub import hf_hub_download

from model import GAT

from explain import (
    create_explainer,
    explain_transaction
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="GAT Fraud Detection",
    page_icon="🔍",
    layout="wide"
)


st.title(
    "🔍 Financial Fraud Detection using GAT"
)

st.write(
    "Enter a transaction ID to obtain a fraud prediction "
    "and GNNExplainer-based explanation."
)


DEVICE = torch.device("cpu")


# ============================================================
# LOAD GRAPH
# ============================================================

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

    return data


data = load_data()


st.success(
    f"Graph loaded: {data.num_nodes:,} transactions"
)


# ============================================================
# LOAD MODEL
# ============================================================

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

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    return model


model = load_model(
    data.num_features
)


st.success(
    "GAT model loaded successfully."
)


# ============================================================
# TRANSACTION ID
# ============================================================

node_id = st.number_input(
    "Enter Transaction ID",

    min_value=0,

    max_value=data.num_nodes - 1,

    value=0,

    step=1
)


# ============================================================
# EXPLAIN
# ============================================================

if st.button(
    "Explain Transaction",
    type="primary"
):

    try:

        with st.spinner(
            "Running GNNExplainer..."
        ):

            explainer = create_explainer(
                model
            )

            result = explain_transaction(
                model,
                data,
                explainer,
                int(node_id)
            )


        # ====================================================
        # PREDICTION
        # ====================================================

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


        # ====================================================
        # LOCAL GRAPH
        # ====================================================

        st.subheader(
            "Local Graph"
        )

        st.write(
            "Local nodes:",
            result["local_nodes"]
        )

        st.write(
            "Local edges:",
            result["local_edges"]
        )


        # ====================================================
        # NEIGHBORS
        # ====================================================

        st.subheader(
            "Top Influential Neighbors"
        )


        if result["neighbors"]:

            for neighbor, importance in result["neighbors"]:

                st.write(
                    f"Transaction **{neighbor}** — "
                    f"Importance: **{importance:.4f}**"
                )
        else:

            st.write(
                "No influential neighbors found."
            )

        # ============================================================
        # GAT ATTENTION
        # ============================================================

        st.subheader(
            "Top GAT Attention Neighbors"
        )

        if result["attention_neighbors"]:

            for neighbor, attention in result["attention_neighbors"]:

                 st.write(
                  f"Transaction **{neighbor}** — "
                  f"GAT Attention: **{attention:.4f}**"
                 )

         else:

                 st.write(
                     "No GAT attention neighbors found."
                 )


        # ====================================================
        # FEATURES
        # ====================================================

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


    except Exception as e:

        st.error(
            "❌ Error while generating explanation"
        )

        st.exception(e)
