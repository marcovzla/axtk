from typing import Optional, Union
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from axtk.token_labeling import LabelingScheme, is_valid_transition


LARGE_NUMBER = 100_000
MISSING_OBSERVATION = -1


def make_start_constraints(id2label: dict[int, str], scheme: LabelingScheme) -> torch.Tensor:
    start_constraints = torch.zeros(len(id2label))
    for i, label in id2label.items():
        if not is_valid_transition(from_label=None, to_label=label, scheme=scheme):
            start_constraints[i] = -torch.inf
    return start_constraints

def make_end_constraints(id2label: dict[int, str], scheme: LabelingScheme) -> torch.Tensor:
    end_constraints = torch.zeros(len(id2label))
    for i, label in id2label.items():
        if not is_valid_transition(from_label=label, to_label=None, scheme=scheme):
            end_constraints[i] = -torch.inf
    return end_constraints

def make_transition_constraints(id2label: dict[int, str], scheme: LabelingScheme) -> torch.Tensor:
    transition_constraints = torch.zeros(len(id2label), len(id2label))
    for i, from_label in id2label.items():
        for j, to_label in id2label.items():
            if not is_valid_transition(from_label=from_label, to_label=to_label, scheme=scheme):
                transition_constraints[i, j] = -torch.inf
    return transition_constraints


def batch_decode(
        emissions: torch.FloatTensor,
        mask: torch.BoolTensor,
        transition_matrix: torch.FloatTensor,
        start_transitions: torch.FloatTensor,
        end_transitions: torch.FloatTensor,
        pad_token_id: int,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """Decodes a batch of sequences and adds padding as needed."""
    # get emissions shape
    batch_size, sequence_length, num_labels = emissions.shape
    # decode label_ids
    batch_label_ids, batch_scores = [], []
    for i in range(batch_size):
        # decode label_ids using viterbi algorithm
        label_ids, scores = decode(
            emissions=emissions[i, mask[i]],
            transition_matrix=transition_matrix,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )
        # pad label_ids to sequence_length
        n_pad = sequence_length - label_ids.shape[1]
        label_ids = F.pad(label_ids, (0, n_pad), value=pad_token_id)
        # store label_ids
        batch_label_ids.append(label_ids)
        batch_scores.append(scores)
    # return decoded labels
    return torch.cat(batch_label_ids), torch.cat(batch_scores)


def decode(
        emissions: torch.FloatTensor,
        transition_matrix: torch.FloatTensor,
        observations: Optional[Sequence[int]] = None,
        start_transitions: Optional[torch.FloatTensor] = None,
        end_transitions: Optional[torch.FloatTensor] = None,
        top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:

    sequence_length, num_labels = emissions.shape
    device = emissions.device

    if top_k < 1:
        raise ValueError('top_k must be >= 1')

    if observations is not None:
        observations = list(observations)
        if len(observations) != sequence_length:
            raise ValueError(f'observations: expected length {sequence_length}, found {len(observations)}')
        for i, observation in enumerate(observations):
            if observation >= num_labels:
                raise ValueError(f'invalid observation at position {i}')
    else:
        observations = [MISSING_OBSERVATION] * sequence_length

    has_start_end_restrictions = start_transitions is not None or end_transitions is not None

    if has_start_end_restrictions:
        # ensure we have both tensors
        start_transitions = ensure_transitions(start_transitions, num_labels).to(device)
        end_transitions = ensure_transitions(end_transitions, num_labels).to(device)
        # merge start and end transitions with transition matrix
        transition_matrix = merge_transitions(transition_matrix, start_transitions, end_transitions)
        # add START and END to observations
        start, end = num_labels, num_labels + 1
        observations = [start] + observations + [end]
        # add START and END to label distributions (impossible)
        emissions = F.pad(emissions, (0, 2), value=-torch.inf)
        # add START and END positions to sequence
        emissions = F.pad(emissions, (0, 0, 1, 1), value=0)

    # execute viterbi algorithm
    viterbi_paths, viterbi_scores = viterbi_algorithm(
        emissions=emissions,
        transition_matrix=transition_matrix,
        observations=observations,
        top_k=top_k,
    )

    # remove START and END from results
    if has_start_end_restrictions:
        viterbi_paths = viterbi_paths[:, 1:-1]

    # return results
    return viterbi_paths, viterbi_scores


def ensure_transitions(
        transitions: Optional[torch.Tensor],
        shape: Union[int, tuple[int], tuple[int, int]],
) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    if transitions is None:
        transitions = torch.zeros(shape)
    elif transitions.shape != shape:
        raise ValueError(f'expected shape {shape}, found {transitions.shape}')
    return transitions


def merge_transitions(
        transition_matrix: torch.FloatTensor,
        start_transitions: torch.FloatTensor,
        end_transitions: torch.FloatTensor,
) -> torch.FloatTensor:
    # pad transitions
    transition_matrix = F.pad(transition_matrix, (0, 2, 0, 2), value=-torch.inf)
    start_transitions = F.pad(start_transitions, (0, 2), value=-torch.inf)
    end_transitions = F.pad(end_transitions, (0, 2), value=-torch.inf)
    # add start and end transitions to transition matrix
    transition_matrix[-2, :] = start_transitions
    transition_matrix[:, -1] = end_transitions
    # return transition matrix
    return transition_matrix


def viterbi_algorithm(
        emissions: torch.FloatTensor,
        transition_matrix: torch.FloatTensor,
        observations: list[int],
        top_k: int,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """Viterbi decoding algorithm in log-space."""

    sequence_length, num_labels = emissions.shape
    device = emissions.device

    path_scores, path_labels = [], []

    # first observation or first tag distribution
    if observations[0] == MISSING_OBSERVATION:
        path_scores.append(emissions[0].unsqueeze(0))
    else:
        one_hot = torch.zeros(1, num_labels, device=device)
        one_hot[0, observations[0]] = LARGE_NUMBER
        path_scores.append(one_hot)

    for i in range(1, sequence_length):
        # add transitions to current scores
        # note the broadcast:
        #   (k, num_labels, 1) + (num_labels, num_labels) = (k, num_labels, num_labels)
        #   k=1 at the beginning, but may grow
        new_scores = path_scores[i-1].unsqueeze(2) + transition_matrix
        new_scores = new_scores.view(-1, num_labels)
        # find best scores for current step and labels for previous step
        k = min(new_scores.shape[0], top_k)
        scores, labels = torch.topk(new_scores, k=k, dim=0)
        # store scores for current step
        if observations[i] == MISSING_OBSERVATION:
            path_scores.append(emissions[i] + scores)
        else:
            one_hot = torch.zeros(1, num_labels, device=device)
            one_hot[0, observations[i]] = LARGE_NUMBER
            path_scores.append(one_hot)
        # store labels for previous step
        path_labels.append(labels.squeeze(0))

    # find best path endings
    complete_path_scores = path_scores[-1].view(-1)
    k = min(complete_path_scores.shape[0], top_k)
    viterbi_scores, best_endings = torch.topk(complete_path_scores, k=k, dim=0)
    viterbi_paths = []

    # reconstruct the best k paths
    for i in range(k):
        # reconstruct path in reverse
        viterbi_path = [best_endings[i]]
        for step_labels in reversed(path_labels):
            step_labels = step_labels.view(-1)
            viterbi_path.append(step_labels[viterbi_path[-1]])
        # reverse reconstructed path
        viterbi_path.reverse()
        # ensure labels are in range [0, num_labels)
        viterbi_path = [label % num_labels for label in viterbi_path]
        # store reconstructed path
        viterbi_paths.append(viterbi_path)

    # return results
    viterbi_paths = torch.tensor(viterbi_paths, device=device)
    return viterbi_paths, viterbi_scores
