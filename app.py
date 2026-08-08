import streamlit as st
import torch

from model import GAT

st.title("GAT Forward Test")

# Small artificial graph
x = torch.randn(10, 166)

edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
], dtype=torch.long)

st.write("Creating GAT model...")

model = GAT(
    in_channels=166,
    hidden_channels=128,
    out_channels=2
)

model.eval()

st.success("Model created")

st.write("Running GAT...")

with torch.no_grad():

    output = model(
        x,
        edge_index
    )

st.success("GAT forward pass works!")

st.write("Output shape:", output.shape)
