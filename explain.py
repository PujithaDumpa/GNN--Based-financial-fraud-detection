import torch

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer


def create_explainer(model):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw"
        )
    )

    return explainer


def explain_transaction(
    model,
    data,
    explainer,
    node_id
):

    model.eval()

    # ---------------------------
    # 1. Prediction
    # ---------------------------

    with torch.no_grad():

        out = model(
            data.x,
            data.edge_index
        )

        probabilities = torch.softmax(
            out,
            dim=1
        )

    prediction = (
        out[node_id]
        .argmax()
        .item()
    )

    confidence = (
        probabilities[node_id][prediction]
        .item()
    )


    # ---------------------------
    # 2. GAT Attention
    # ---------------------------

    with torch.no_grad():

        (
            out,
            edge_index1,
            att1,
            edge_index2,
            att2
        ) = model(
            data.x,
            data.edge_index,
            return_attention=True
        )


    edge_index = edge_index2

    attention_weights = att2.squeeze()


    # Find edges connected to node

    mask = (
        (edge_index[0] == node_id) |
        (edge_index[1] == node_id)
    )


    connected_edges = edge_index[:, mask]

    connected_attention = (
        attention_weights[mask]
    )


    neighbors = []


    for i in range(
        connected_edges.shape[1]
    ):

        source = (
            connected_edges[0, i]
            .item()
        )

        target = (
            connected_edges[1, i]
            .item()
        )

        weight = (
            connected_attention[i]
            .item()
        )


        if source == node_id:

            neighbor = target

        else:

            neighbor = source


        # Remove self-loop

        if neighbor != node_id:

            neighbors.append(
                (neighbor, weight)
            )


    neighbors = sorted(
        neighbors,
        key=lambda x: x[1],
        reverse=True
    )


    top_neighbors = neighbors[:5]


    # ---------------------------
    # 3. Feature Importance
    # ---------------------------

    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_id
    )


    feature_importance = (
        explanation.node_mask[node_id]
    )


    top_values, top_indices = (
        torch.topk(
            feature_importance,
            k=10
        )
    )


    important_features = []


    for feature, score in zip(
        top_indices,
        top_values
    ):

        important_features.append(
            (
                feature.item(),
                score.item()
            )
        )


    # ---------------------------
    # 4. Return
    # ---------------------------

    return {

        "node_id": node_id,

        "prediction": prediction,

        "confidence": confidence,

        "neighbors": top_neighbors,

        "features": important_features
    }