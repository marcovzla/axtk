from typing import Optional


def find_context_positions(
        sequence_ids: Optional[list[int]] = None,
        offset_mapping: Optional[list[tuple[int, int]]] = None,
) -> tuple[int, int]:
    if sequence_ids is not None:
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        return context_start, context_end
    elif offset_mapping is not None:
        context_start = 0
        while offset_mapping[context_start] == (0, 0):
            context_start += 1
        context_end = len(offset_mapping) - 1
        while offset_mapping[context_end] == (0, 0):
            context_end -= 1
        return context_start, context_end
    else:
        raise ValueError('either sequence_ids or offset_mapping must be specified')


def find_token_positions(
        offset_mapping: list[tuple[int, int]],
        start_char: int,
        stop_char: int,
        *,
        context: Optional[tuple[int, int]] = None,
        sequence_ids: Optional[list[int]] = None,
) -> tuple[int, int]:
    # if context positions were not given, find them
    if context is None:
        context = find_context_positions(sequence_ids=sequence_ids, offset_mapping=offset_mapping)
    # get context start and end
    context_start, context_end = context
    # check if character span is fully inside context
    if offset_mapping[context_start][0] > start_char or offset_mapping[context_end][1] < stop_char:
        return 0, 0
    # find index of first token in span
    idx = context_start
    while idx <= context_end and offset_mapping[idx][0] <= start_char:
        idx += 1
    start_position = idx - 1
    # find index of last token in span
    idx = context_end
    while idx >= context_start and offset_mapping[idx][1] >= stop_char:
        idx -= 1
    end_position = idx + 1
    # return positions of first and last token in span
    return start_position, end_position
