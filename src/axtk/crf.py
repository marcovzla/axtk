from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from axtk import viterbi
from axtk.torch_utils import defrag, get_device


class LinearChainCRF(nn.Module):
    def __init__(
            self,
            num_labels: int,
            transition_constraints: Optional[torch.Tensor] = None,
            start_constraints: Optional[torch.Tensor] = None,
            end_constraints: Optional[torch.Tensor] = None,
            ignore_index: int = -100,
    ):
        super().__init__()
        self.num_labels = num_labels + 2
        self.ignore_index = ignore_index
        # define start and end label indices
        self.start = self.num_labels - 2
        self.end = self.num_labels - 1
        # initialize trainable transition matrix
        self.transition_matrix = nn.Parameter(torch.empty(self.num_labels, self.num_labels))
        nn.init.xavier_normal_(self.transition_matrix)
        # ensure constraints are provided and in the correct shape
        transition_constraints = viterbi.ensure_transitions(transition_constraints, (num_labels, num_labels))
        start_constraints = viterbi.ensure_transitions(start_constraints, num_labels)
        end_constraints = viterbi.ensure_transitions(end_constraints, num_labels)
        # merge constraints
        transition_constraints = viterbi.merge_transitions(
            transition_matrix=transition_constraints,
            start_transitions=start_constraints,
            end_transitions=end_constraints,
        )
        # store constraints in buffer
        self.register_buffer('transition_constraints', transition_constraints)

    def get_transitions(self, apply_constraints: bool = False):
        transitions = self.transition_matrix
        # apply constraints
        if apply_constraints:
            transitions = transitions + self.transition_constraints
        # log-softmax
        transitions = F.log_softmax(transitions, dim=1)
        # separate start and end transitions
        transition_matrix = transitions[:-2, :-2]
        start_transitions = transitions[self.start, :-2]
        end_transitions = transitions[:-2, self.end]
        # return results
        return transition_matrix, start_transitions, end_transitions

    def forward(
            self,
            *,
            labels: torch.LongTensor,
            logits: Optional[torch.Tensor] = None,
            emissions: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the negative log-likelihoods for each batch element."""
        return -self.log_likelihoods(
            logits=logits,
            emissions=emissions,
            labels=labels,
            mask=mask,
        )

    def log_likelihoods(
            self,
            *,
            labels: torch.LongTensor,
            logits: Optional[torch.Tensor] = None,
            emissions: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the log-likelihood for each sequence of emissions and their corresponding labels."""
        # ensure we have emission log-probabilities
        if emissions is None and logits is None:
            raise ValueError('either emissions or logits must be provided')
        elif emissions is not None and logits is not None:
            raise ValueError('either emissions or logits must be provided, but not both')
        elif emissions is None:
            emissions = F.log_softmax(logits, dim=-1)
        # ensure we have a boolean mask
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            mask = mask.type(torch.bool)
        # ignore labels if set to ignore_index
        mask[labels == self.ignore_index] = False
        # defrag input tensors
        emissions = defrag(emissions, mask)
        labels = defrag(labels, mask, empty_value=0)
        mask = defrag(mask, mask, empty_value=False)
        # transitions log-probabilities
        transitions, start_transitions, end_transitions = self.get_transitions()
        # return log-likelihood
        return log_likelihoods(
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
            emissions=emissions,
            labels=labels,
            mask=mask,
        )

    @torch.no_grad()
    def batch_decode(
            self,
            *,
            logits: Optional[torch.Tensor] = None,
            emissions: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        if device is None:
            device = get_device(self)
        # ensure we have emission log-probabilities
        if emissions is None and logits is None:
            raise ValueError('either emissions or logits must be provided')
        elif emissions is not None and logits is not None:
            raise ValueError('either emissions or logits must be provided, but not both')
        elif emissions is None:
            emissions = F.log_softmax(logits, dim=-1)
        # ensure we have a boolean mask
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool)
        else:
            mask = mask.type(torch.bool)
        # defrag input tensors
        emissions = defrag(emissions, mask)
        mask = defrag(mask, mask, empty_value=False)
        # transitions log-probabilities
        transitions, start_transitions, end_transitions = self.get_transitions(apply_constraints=True)
        # decode labels
        return viterbi.batch_decode(
            emissions=emissions.to(device),
            mask=mask.to(device),
            transition_matrix=transitions.to(device),
            start_transitions=start_transitions.to(device),
            end_transitions=end_transitions.to(device),
            pad_token_id=self.ignore_index,
        )


def log_likelihoods(
        transitions: torch.Tensor,
        start_transitions: torch.Tensor,
        end_transitions: torch.Tensor,
        emissions: torch.Tensor,
        labels: torch.LongTensor,
        mask: torch.BoolTensor,
) -> torch.Tensor:
    """Computes the log-likelihood for each sequence of emissions and their corresponding labels."""
    # compute numerator
    scores = compute_scores(
        transitions=transitions,
        start_transitions=start_transitions,
        end_transitions=end_transitions,
        emissions=emissions,
        labels=labels,
        mask=mask,
    )
    # compute denominator
    partitions = compute_partitions(
        transitions=transitions,
        start_transitions=start_transitions,
        end_transitions=end_transitions,
        emissions=emissions,
        mask=mask,
    )
    # return log-likelihoods
    return scores - partitions


def compute_scores(
        transitions: torch.Tensor,
        start_transitions: torch.Tensor,
        end_transitions: torch.Tensor,
        emissions: torch.Tensor,
        labels: torch.LongTensor,
        mask: torch.BoolTensor,
) -> torch.Tensor:
    """Computes the scores for a given batch of emissions and their corresponding labels."""
    batch_size, sequence_length, num_labels = emissions.shape
    # compute transition and emmission scores for first labels in batch
    first_labels = labels[:, 0]
    t_scores = start_transitions[first_labels]
    e_scores = emissions[:, 0].gather(1, first_labels.unsqueeze(1)).squeeze(1)
    # store current scores
    scores = t_scores + e_scores
    # accumulate scores for remaining labels
    for i in range(sequence_length):
        previous_labels = labels[:, i-1]
        current_labels = labels[:, i]
        # compute transition and emission scores
        t_scores = transitions[previous_labels, current_labels]
        e_scores = emissions[:, i].gather(1, current_labels.unsqueeze(1)).squeeze(1)
        # accumulate scores
        scores += mask[:, i] * (t_scores + e_scores)
    # add transition score from last label to END
    last_label_indices = mask.sum(1) - 1
    last_labels = labels.gather(1, last_label_indices.unsqueeze(1)).squeeze(1)
    scores += end_transitions[last_labels]
    # return score
    return scores


def compute_partitions(
        transitions: torch.Tensor,
        start_transitions: torch.Tensor,
        end_transitions: torch.Tensor,
        emissions: torch.Tensor,
        mask: torch.BoolTensor,
) -> torch.Tensor:
    """Computes the partition function for each batch element using the forward-algorithm."""
    # see https://www.youtube.com/watch?v=fGdXkVv1qNQ
    batch_size, sequence_length, num_labels = emissions.shape
    # alphas shape (batch_size, num_labels)
    alphas = start_transitions.unsqueeze(0) + emissions[:, 0]
    # accumulate alphas from left to right
    for i in range(1, sequence_length):
        next_alphas = torch.logsumexp(
            # broadcast transition scores over batch dimension
            transitions.unsqueeze(0)
            # broadcast emission scores over previous_label dimension
            + emissions[:, i].unsqueeze(1)
            # broadcast alphas over current_label dimension
            + alphas.unsqueeze(2),
            # logsumexp over previous_label dimension
            dim=1,
        )
        # only update alphas if mask is true
        step_mask = mask[:, i].unsqueeze(1)
        alphas = step_mask * next_alphas + ~step_mask * alphas
    # add transition scores to END
    alphas += end_transitions.unsqueeze(0)
    # return partition values
    return torch.logsumexp(alphas, dim=1)
