import streamlit as st
import torch
from huggingface_hub import hf_hub_download

st.title("Graph Loading Test")

st.write("Downloading graph...")

data_path = hf_hub_download(
    repo_id="PujithaDumpa/elliptic-gat-data",
    filename="elliptic_data.pt",
    repo_type="model"
)

st.success("Download successful!")

st.write("Loading graph into memory...")

data = torch.load(
    data_path,
    map_location="cpu",
    weights_only=False
)

st.success("Graph loaded successfully!")

st.write("Number of nodes:", data.num_nodes)
st.write("Number of features:", data.num_features)
st.write("Number of edges:", data.edge_index.shape[1])
