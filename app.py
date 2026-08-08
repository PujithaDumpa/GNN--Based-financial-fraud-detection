import streamlit as st
import torch

from huggingface_hub import hf_hub_download
from torch_geometric.utils import k_hop_subgraph

from model import GAT


st.title("Local GAT Prediction Test")

DEVICE = torch.device("cpu")


# =========================
# LOAD GRAPH
# =========================

st.write("Loading graph...")

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

st.success("Graph loaded")


# =========================
# LOAD MODEL
# =========================

model = GAT(
    in_channels=data.num_features,
    hidden_channels=128,
    out_channels=2
)

state_dict = torch.load(
    "best_gat_model.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()

st.success("GAT model loaded")


# =========================
# SELECT TRANSACTION
# =========================

node_id = st.number_input(
    "Transaction ID",
    min_value=0,
    max_value=data.num_nodes - 1,
    value=0,
    step=1
)


# =========================
# LOCAL PREDICTION
# =========================

if st.button("Test Prediction"):

    node_id = int(node_id)

    st.write("Creating local neighborhood...")

    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_id,
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=True
    )

    sub_x = data.x[subset]

    local_node_id = mapping.item()

    st.write("Local nodes:", sub_x.size(0))
    st.write("Local edges:", sub_edge_index.size(1))

    st.write("Running GAT on local graph...")

    with torch.no_grad():

        output = model(
            sub_x,
            sub_edge_index
        )

    prediction = output[
        local_node_id
    ].argmax().item()

    probability = torch.softmax(
        output[local_node_id],
        dim=0
    )

    st.success("Prediction completed!")

    st.write("Prediction:", prediction)

    st.write(
        "Confidence:",
        probability[prediction].item()
    )
