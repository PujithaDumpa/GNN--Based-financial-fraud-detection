import streamlit as st
import torch
from huggingface_hub import hf_hub_download

st.title("GAT Fraud Detection")

st.write("App started")

DEVICE = torch.device("cpu")

st.write("Trying to load graph...")

data_path = hf_hub_download(
    repo_id="PujithaDumpa/elliptic-gat-data",
    filename="elliptic_data.pt",
    repo_type="model"
)

st.write("Graph file downloaded")

data = torch.load(
    data_path,
    map_location=DEVICE,
    weights_only=False
)

st.success("Graph loaded successfully!")

st.write("Number of transactions:", data.num_nodes)
st.write("Number of features:", data.num_features)
