import streamlit as st
import torch
from huggingface_hub import hf_hub_download

st.title("Dataset Test")

st.write("Streamlit: ✅")
st.write("PyTorch:", torch.__version__)

st.write("Downloading graph...")

data_path = hf_hub_download(
    repo_id="PujithaDumpa/elliptic-gat-data",
    filename="elliptic_data.pt",
    repo_type="model"
)

st.success("Graph file downloaded successfully!")

st.write("File path:", data_path)
