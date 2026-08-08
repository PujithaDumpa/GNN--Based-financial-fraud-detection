
import streamlit as st
import torch
import pandas as pd

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
st.write(data)
# ============================================================
# LOAD ORIGINAL ELLIPTIC TRANSACTION IDs
# ============================================================

classes = pd.read_csv("elliptic_txs_classes.csv")

if len(classes) != data.num_nodes:

    st.error(
        f"Dataset mismatch: classes.csv contains "
        f"{len(classes)} transactions, but the graph contains "
        f"{data.num_nodes} nodes."
    )

    st.stop()


# Convert transaction IDs to strings
classes["txId"] = classes["txId"].astype(str)


# Map actual transaction ID → PyG node index
txid_to_node = {
    txid: index
    for index, txid in enumerate(classes["txId"])
}


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

transaction_id = st.text_input(
    "Enter Elliptic Transaction ID",
    placeholder="Example: 232438397"
)


# ============================================================
# EXPLAIN
# ============================================================
if st.button(
    "Explain Transaction",
    type="primary"
):

    transaction_id = transaction_id.strip()

    if transaction_id == "":

        st.warning(
            "Please enter a transaction ID."
        )

        st.stop()


    if transaction_id not in txid_to_node:

        st.error(
            "Transaction ID not found in the Elliptic dataset."
        )

        st.stop()


    # Convert actual txId → internal node index

    node_id = txid_to_node[transaction_id]


    st.write(
        f"Transaction ID: **{transaction_id}**"
    )

    st.write(
        f"Internal node index: **{node_id}**"
    )


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
                node_id
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
        # GNNEXPLAINER NEIGHBORS
        # ====================================================

        st.subheader(
            "Top Influential Neighbors"
        )

        if result["neighbors"]:

            for neighbor, importance in result["neighbors"]:

                st.write(
                    f"Node **{neighbor}** — "
                    f"GNNExplainer Importance: "
                    f"**{importance:.4f}**"
                )

        else:

            st.write(
                "No influential neighbors found."
            )


        # ====================================================
        # IMPORTANT FEATURES
        # ====================================================

        st.subheader(
            "Top Important Features"
        )

        if result["features"]:

            for txid, importance in result["neighbors"]:

               st.write(
                  f"Transaction **{txid}** — "
                  f"GNNExplainer Importance: **{importance:.4f}**"
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
