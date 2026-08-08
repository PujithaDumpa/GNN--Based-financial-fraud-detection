import torch

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.utils import k_hop_subgraph


# ============================================================
# CREATE GNN EXPLAINER
# ============================================================

def create_explainer(model):

    return Explainer(
        model=model,

        algorithm=GNNExplainer(
            epochs=20
        ),

        explanation_type="model",

        node_mask_type="attributes",

        edge_mask_type="object",

        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw"
        )
    )


# ============================================================
# EXPLAIN TRANSACTION
# ============================================================

def explain_transaction(
    model,
    data,
    explainer,
    node_id
):

    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # ========================================================
    # 1. PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            x,
            edge_index
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = (
            output[node_id]
            .argmax()
            .item()
        )

        confidence = (
            probabilities[
                node_id,
                prediction
            ].item()
        )

    # ========================================================
    # 2. CREATE LOCAL 2-HOP SUBGRAPH
    # ========================================================

    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_id,
        num_hops=2,
        edge_index=edge_index,
        relabel_nodes=True
    )

    sub_x = x[subset]
    print("Subgraph nodes:", sub_x.size(0))
    print("Subgraph edges:", sub_edge_index.size(1))

    local_node_id = mapping.item()

    # ========================================================
    # 3. RUN GNN EXPLAINER
    # ========================================================

    explanation = explainer(
        x=sub_x,
        edge_index=sub_edge_index,
        index=local_node_id
    )

    # ========================================================
    # 4. FEATURE IMPORTANCE
    # ========================================================

    node_mask = explanation.node_mask

    if node_mask is None:
        important_features = []

    else:

        if node_mask.dim() == 2:

            feature_importance = node_mask[
                local_node_id
            ]

        else:

            feature_importance = node_mask

        feature_importance = feature_importance.abs()

        k = min(
            10,
            feature_importance.numel()
        )

        top_values, top_indices = torch.topk(
            feature_importance,
            k=k
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

    # ========================================================
    # 5. IMPORTANT NEIGHBORS
    # ========================================================

    neighbors = []

    explanation_edge_mask = explanation.edge_mask

    if explanation_edge_mask is not None:

        edge_k = min(
            20,
            explanation_edge_mask.numel()
        )

        edge_values, edge_indices = torch.topk(
            explanation_edge_mask,
            k=edge_k
        )

        for edge_idx, importance in zip(
            edge_indices,
            edge_values
        ):

            source = sub_edge_index[
                0,
                edge_idx
            ].item()

            target = sub_edge_index[
                1,
                edge_idx
            ].item()

            # Only edges connected to selected node

            if source == local_node_id:

                neighbor_local = target

            elif target == local_node_id:

                neighbor_local = source

            else:

                continue

            neighbor_original = subset[
                neighbor_local
            ].item()

            # Remove self-loop

            if neighbor_original == node_id:
                continue

            neighbors.append(
                (
                    neighbor_original,
                    importance.item()
                )
            )

    # ========================================================
    # 6. SORT AND REMOVE DUPLICATES
    # ========================================================

    neighbors = sorted(
        neighbors,
        key=lambda x: x[1],
        reverse=True
    )

    unique_neighbors = []

    seen = set()

    for neighbor, importance in neighbors:

        if neighbor not in seen:

            unique_neighbors.append(
                (
                    neighbor,
                    importance
                )
            )

            seen.add(neighbor)

    top_neighbors = unique_neighbors[:5]

    # ========================================================
    # 7. RETURN RESULTS
    # ========================================================

    return {

        "node_id": node_id,

        "prediction": prediction,

        "confidence": confidence,

        "neighbors": top_neighbors,

        "features": important_features
    }
