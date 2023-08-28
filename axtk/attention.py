from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralAttention(nn.Module):
    """https://arxiv.org/pdf/1508.04025.pdf"""

    def __init__(self, target_embedding_size: int, source_embedding_size: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(target_embedding_size, source_embedding_size))
        nn.init.xavier_uniform_(self.W)

    def forward(
            self,
            target_vectors: torch.Tensor,
            source_vectors: torch.Tensor,
            source_mask: Optional[torch.Tensor] = None,
    ):
        # target_vectors: (batch_size, target_embedding_size)
        # source_vectors: (batch_size, sequence_length, source_embedding_size)
        output = target_vectors @ self.W
        output = source_vectors @ output.unsqueeze(2)
        output = output.squeeze()
        if source_mask is not None:
            output = output * source_mask
        return output


class ConcatAttention(nn.Module):
    """https://arxiv.org/pdf/1508.04025.pdf"""

    def __init__(self, target_embedding_size: int, source_embedding_size: int, hidden_size: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(target_embedding_size + source_embedding_size, hidden_size))
        nn.init.xavier_uniform_(self.W)
        self.v = nn.Parameter(torch.empty(hidden_size))
        nn.init.xavier_uniform_(self.v)

    def forward(
            self,
            target_vectors: torch.Tensor,
            source_vectors: torch.Tensor,
            source_mask: Optional[torch.Tensor] = None,
    ):
        # target_vectors: (batch_size, target_embedding_size)
        # source_vectors: (batch_size, sequence_length, source_embedding_size)
        seq_len = source_vectors.size(dim=1)
        target_vectors = target_vectors.unsqueeze(1).repeat(1, seq_len, 1)
        concat_vectors = torch.cat([source_vectors, target_vectors], dim=2)
        output = F.tanh(concat_vectors @ self.W) @ self.v
        if source_mask is not None:
            output = output * source_mask
        return output
