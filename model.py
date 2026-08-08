import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv


class GAT(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=4
        )

        self.bn1 = BatchNorm1d(
            hidden_channels * 4
        )

        self.conv2 = GATConv(
            hidden_channels * 4,
            out_channels,
            heads=1
        )


    def forward(self, x, edge_index, return_attention=False):

        if return_attention:

            # First GAT layer + attention
            x, (edge_index1, att1) = self.conv1(
                x,
                edge_index,
                return_attention_weights=True
            )

            x = self.bn1(x)

            x = F.relu(x)

            x = F.dropout(
                x,
                p=0.3,
                training=self.training
            )


            # Second GAT layer + attention
            x, (edge_index2, att2) = self.conv2(
                x,
                edge_index,
                return_attention_weights=True
            )

            return x, edge_index1, att1, edge_index2, att2


        else:

            # Normal training/inference output
            x = self.conv1(
                x,
                edge_index
            )

            x = self.bn1(x)

            x = F.relu(x)

            x = F.dropout(
                x,
                p=0.3,
                training=self.training
            )

            x = self.conv2(
                x,
                edge_index
            )

            return x