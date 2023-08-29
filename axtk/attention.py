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
            source_vectors: torch.Tensor,
            target_vectors: torch.Tensor,
            source_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
    ):
        # source_vectors: (batch_size, source_sequence_length, source_embedding_size)
        # target_vectors: (batch_size, target_embedding_size) or (batch_size, target_sequence_length, target_embedding_size)
        target_is_single_vector = len(target_vectors.shape) == 2
        if target_is_single_vector:
            if target_mask is not None:
                raise ValueError('target_mask is not valid when target is a single vector')
            target_vectors = target_vectors.unsqueeze(1)
        # output: (batch_size, target_sequence_length, source_sequence_length)
        output = target_vectors @ self.W @ source_vectors.transpose(1, 2)
        output = apply_masks(output, source_mask, target_mask)
        # squeeze target_sequenze_length if target is a single vector
        if target_is_single_vector:
            output = output.squeeze(1)
        # return results
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
            source_vectors: torch.Tensor,
            target_vectors: torch.Tensor,
            source_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
    ):
        # source_vectors: (batch_size, sequence_length, source_embedding_size)
        # target_vectors: (batch_size, target_embedding_size) or (batch_size, target_sequence_length, target_embedding_size)
        target_is_single_vector = len(target_vectors.shape) == 2
        if target_is_single_vector:
            if target_mask is not None:
                raise ValueError('target_mask is not valid when target is a single vector')
            target_vectors = target_vectors.unsqueeze(1)
        src_seq_len = source_vectors.size(dim=1)
        tgt_seq_len = target_vectors.size(dim=1)
        source_vectors = source_vectors.unsqueeze(2).repeat(1, tgt_seq_len, 1, 1)
        target_vectors = target_vectors.unsqueeze(1).repeat(1, 1, src_seq_len, 1)
        # concat_vectors: (batch_size, target_sequence_length, source_sequence_length, source_embedding_size + target_embedding_size)
        concat_vectors = torch.cat([source_vectors, target_vectors], dim=3)
        # output: (batch_size, target_sequence_length, source_sequence_length)
        output = F.tanh(concat_vectors @ self.W) @ self.v
        output = apply_masks(output, source_mask, target_mask)
        # squeeze target_sequenze_length if target is a single vector
        if target_is_single_vector:
            output = output.squeeze(1)
        # return results
        return output


class BiaffineAttention(nn.Module):
    """https://arxiv.org/pdf/1611.01734.pdf"""

    def __init__(self, target_embedding_size: int, source_embedding_size: int, target_hidden_size: int, souce_hidden_size: int):
        super().__init__()
        self.W_src = nn.Parameter(torch.empty(source_embedding_size, souce_hidden_size))
        nn.init.xavier_uniform_(self.W_src)
        self.W_tgt = nn.Parameter(torch.empty(target_embedding_size, target_hidden_size))
        nn.init.xavier_uniform_(self.W_tgt)
        self.U = nn.Parameter(torch.empty(target_hidden_size, souce_hidden_size))
        nn.init.xavier_uniform_(self.U)
        self.u = nn.Parameter(torch.empty(souce_hidden_size))
        nn.init.xavier_uniform_(self.u)

    def forward(
            self,
            source_vectors: torch.Tensor,
            target_vectors: torch.Tensor,
            source_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
    ):
        # source_vectors: (batch_size, source_sequence_length, source_embedding_size)
        # target_vectors: (batch_size, target_embedding_size) or (batch_size, target_sequence_length, target_embedding_size)
        # source_mask: (batch_size, source_sequence_length)
        # target_mask: (batch_size, target_sequence_length)
        target_is_single_vector = len(target_vectors.shape) == 2
        if target_is_single_vector:
            if target_mask is not None:
                raise ValueError('target_mask is not valid when target is a single vector')
            target_vectors = target_vectors.unsqueeze(1)
        # source_output: (batch_size, 1, source_sequence_length)
        source_output = (source_vectors @ self.W_src).unsqueeze(1)
        # target_output: (batch_size, target_sequence_length, 1)
        target_output = (target_vectors @ self.W_tgt).unsqueeze(2)
        # output: (batch_size, target_sequence_length, source_sequence_length)
        output = target_output @ self.U @ source_output.transpose(1, 2)
        output = output + source_output @ self.u
        output = apply_masks(output, source_mask, target_mask)
        # squeeze target_sequenze_length if target is a single vector
        if target_is_single_vector:
            output = output.squeeze(1)
        # return results
        return output


def apply_masks(
        scores: torch.Tensor,
        source_mask: Optional[torch.Tensor],
        target_mask: Optional[torch.Tensor],
        masked_value: float = -torch.inf,
) -> torch.Tensor:
    # get shape
    batch_size, target_len, source_len = scores.size()
    # apply source_mask
    if source_mask is not None:
        mask = source_mask.unsqueeze(1).repeat(1, target_len, 1)
        scores[mask==0] = masked_value
    # apply target_mask
    if target_mask is not None:
        mask = target_mask.unsqueeze(2).repeat(1, 1, source_len)
        scores[mask==0] = masked_value
    # return scores
    return scores
