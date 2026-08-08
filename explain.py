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
            epochs=150
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
    # 1. CREATE LOCAL 7-HOP SUBGRAPH
    # ========================================================

    subset, sub_edge_index, mapping, _ = k_hop_subgraph(

        node_id,

        num_hops=7,

        edge_index=edge_index,

        relabel_nodes=True
    )

    sub_x = x[subset]

    local_node_id = mapping.item()


    # ========================================================
    # 2. PREDICTION ON LOCAL GRAPH
    # ========================================================

    with torch.no_grad():

        output = model(
            sub_x,
            sub_edge_index
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = (
            output[
                local_node_id
            ]
            .argmax()
            .item()
        )

        confidence = (
            probabilities[
                local_node_id,
                prediction
            ]
            .item()
        )


    # ========================================================
    # 3. RUN GNNEXPLAINER
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

    important_features = []


    if node_mask is not None:

        if node_mask.dim() == 2:

            feature_importance = (
                node_mask[
                    local_node_id
                ]
            )

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
    # 5. IMPORTANT NEIGHBORS
    # ========================================================

    neighbors = {}

    explanation_edge_mask = (
        explanation.edge_mask
    )


    if explanation_edge_mask is not None:

        # Number of edges available in both tensors

        num_edges = min(
            explanation_edge_mask.numel(),
            sub_edge_index.size(1)
        )


        for edge_idx in range(
            num_edges
        ):

            importance = (
                explanation_edge_mask[
                    edge_idx
                ]
                .item()
            )


            source = (
                sub_edge_index[
                    0,
                    edge_idx
                ]
                .item()
            )


            target = (
                sub_edge_index[
                    1,
                    edge_idx
                ]
                .item()
            )


            # ------------------------------------------------
            # Check if edge touches the selected transaction
            # ------------------------------------------------

            if source == local_node_id:

                neighbor_local = target

            elif target == local_node_id:

                neighbor_local = source

            else:

                continue


            # ------------------------------------------------
            # Convert local node → original graph node
            # ------------------------------------------------

            neighbor_original = (
                subset[
                    neighbor_local
                ]
                .item()
            )


            # ------------------------------------------------
            # Remove self-loop
            # ------------------------------------------------

            if neighbor_original == node_id:

                continue


            # ------------------------------------------------
            # Keep highest importance for each neighbor
            # ------------------------------------------------

            if (
                neighbor_original not in neighbors
                or
                importance >
                neighbors[
                    neighbor_original
                ]
            ):

                neighbors[
                    neighbor_original
                ] = importance


    # ========================================================
    # 6. SORT NEIGHBORS
    # ========================================================

    top_neighbors = sorted(

        neighbors.items(),

        key=lambda x: x[1],

        reverse=True

    )[:5]


    # ========================================================
    # 7. RETURN RESULTS
    # ========================================================

    return {

        "node_id": node_id,

        "prediction": prediction,

        "confidence": confidence,

        "neighbors": top_neighbors,

        "features": important_features,

        "local_nodes": sub_x.size(0),

        "local_edges": sub_edge_index.size(1)

    }
