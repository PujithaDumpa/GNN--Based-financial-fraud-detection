import torch

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer


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

    # ========================================================
    # FULL GRAPH
    # ========================================================

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

        prediction = output[
            node_id
        ].argmax().item()

        confidence = probabilities[
            node_id,
            prediction
        ].item()


    # ========================================================
    # 2. RUN GNNEXPLAINER ON FULL GRAPH
    # ========================================================

    explanation = explainer(
        x=x,
        edge_index=edge_index,
        index=node_id
    )


    # ========================================================
    # 3. FEATURE IMPORTANCE
    # ========================================================

    node_mask = explanation.node_mask

    important_features = []


    if node_mask is not None:

        if node_mask.dim() == 2:

            feature_importance = node_mask[
                node_id
            ]

        else:

            feature_importance = node_mask


        feature_importance = (
            feature_importance.abs()
        )


        k = min(
            5,
            feature_importance.numel()
        )


        top_values, top_indices = torch.topk(
            feature_importance,
            k=k
        )


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
    # 4. IMPORTANT NEIGHBORS
    # ========================================================

    neighbors = []

    explanation_edge_mask = (
        explanation.edge_mask
    )


    if explanation_edge_mask is not None:

        edge_k = min(
            50,
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

            source = edge_index[
                0,
                edge_idx
            ].item()


            target = edge_index[
                1,
                edge_idx
            ].item()


            # Only edges connected to
            # the selected transaction.

            if source == node_id:

                neighbor = target

            elif target == node_id:

                neighbor = source

            else:

                continue


            # Remove self-loop.

            if neighbor == node_id:

                continue


            neighbors.append(
                (
                    neighbor,
                    importance.item()
                )
            )


    # ========================================================
    # 5. SORT NEIGHBORS
    # ========================================================

    neighbors.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # 6. REMOVE DUPLICATES
    # ========================================================

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
