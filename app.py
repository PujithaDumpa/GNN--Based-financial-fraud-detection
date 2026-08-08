import streamlit as st
import torch

from huggingface_hub import hf_hub_download

from model import GAT


st.title("GAT Model Test")

DEVICE = torch.device("cpu")


# Load graph

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


# Load GAT

st.write("Loading GAT model...")

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

st.success("GAT model loaded successfully!")
