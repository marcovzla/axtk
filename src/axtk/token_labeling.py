from dataclasses import dataclass
from typing import Optional, Union, Literal
from enum import Enum
from axtk.span import Span



class Scheme(str, Enum):
    IO = 'IO'
    IOB1 = 'IOB1'
    IOB2 = 'IOB2'
    BILOU = 'BILOU'
    IOBES = 'IOBES'
    RAW = 'RAW'

LabelingScheme = Union[Scheme, Literal['IO', 'IOB1', 'IOB2', 'BILOU', 'IOBES', 'RAW']]



@dataclass(frozen=True, order=True)
class TokenSpan(Span):
    label: Optional[str]



def is_valid_label(label: str, scheme: LabelingScheme) -> bool:
    scheme = Scheme(scheme)
    if scheme == Scheme.IO:
        return io_valid_label(label)
    elif scheme == Scheme.IOB1:
        return iob1_valid_label(label)
    elif scheme == Scheme.IOB2:
        return iob2_valid_label(label)
    elif scheme == Scheme.BILOU:
        return bilou_valid_label(label)
    elif scheme == Scheme.IOBES:
        return iobes_valid_label(label)
    elif scheme == Scheme.RAW:
        return raw_valid_label(label)
    else:
        raise ValueError(f'invalid {scheme=}')



def is_valid_transition(from_label: Optional[str], to_label: Optional[str], scheme: LabelingScheme) -> bool:
    scheme = Scheme(scheme)
    if scheme == Scheme.IO:
        return io_valid_transition(from_label, to_label)
    elif scheme == Scheme.IOB1:
        return iob1_valid_transition(from_label, to_label)
    elif scheme == Scheme.IOB2:
        return iob2_valid_transition(from_label, to_label)
    elif scheme == Scheme.BILOU:
        return bilou_valid_transition(from_label, to_label)
    elif scheme == Scheme.IOBES:
        return iobes_valid_transition(from_label, to_label)
    elif scheme == Scheme.RAW:
        return raw_valid_transition(from_label, to_label)
    else:
        raise ValueError(f'invalid {scheme=}')



def load_spans(labels: list[str], scheme: LabelingScheme) -> list[TokenSpan]:
    scheme = Scheme(scheme)
    if scheme == Scheme.IO:
        return io_to_spans(labels)
    elif scheme == Scheme.IOB1:
        return iob1_to_spans(labels)
    elif scheme == Scheme.IOB2:
        return iob2_to_spans(labels)
    elif scheme == Scheme.BILOU:
        return bilou_to_spans(labels)
    elif scheme == Scheme.IOBES:
        return iobes_to_spans(labels)
    elif scheme == Scheme.RAW:
        return raw_to_spans(labels)
    else:
        raise ValueError(f'invalid {scheme=}')



def dump_spans(num_tokens: int, spans: list[TokenSpan], scheme: LabelingScheme) -> list[str]:
    scheme = Scheme(scheme)
    if scheme == Scheme.IO:
        return spans_to_io(num_tokens, spans)
    elif scheme == Scheme.IOB1:
        return spans_to_iob1(num_tokens, spans)
    elif scheme == Scheme.IOB2:
        return spans_to_iob2(num_tokens, spans)
    elif scheme == Scheme.BILOU:
        return spans_to_bilou(num_tokens, spans)
    elif scheme == Scheme.IOBES:
        return spans_to_iobes(num_tokens, spans)
    elif scheme == Scheme.RAW:
        return spans_to_raw(num_tokens, spans)
    else:
        raise ValueError(f'invalid {scheme=}')



def convert_labels(labels: list[str], from_scheme: LabelingScheme, to_scheme: LabelingScheme) -> list[str]:
    return dump_spans(len(labels), load_spans(labels, from_scheme), to_scheme)



def parse_label(label: str, sep: str = '-') -> tuple[Optional[str], Optional[str]]:
    if sep in label:
        return tuple(label.split(sep=sep, maxsplit=1))
    elif label in 'OBILUES':
        return label, None
    else:
        return None, label



def raw_valid_label(label: str) -> bool:
    return bool(label)

def io_valid_label(label: str) -> bool:
    return label and label[0] in 'IO'

def iob1_valid_label(label: str) -> bool:
    return label and label[0] in 'BIO'

def iob2_valid_label(label: str) -> bool:
    return label and label[0] in 'BIO'

def bilou_valid_label(label: str) -> bool:
    return label and label[0] in 'BILUO'

def iobes_valid_label(label: str) -> bool:
    return label and label[0] in 'IOBES'



def raw_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    return from_label is not None or to_label is not None

def io_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    return from_label is not None or to_label is not None

def iob1_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    if from_label is None:
        return to_label is not None and to_label[0] in 'IO'
    if to_label is None:
        return True
    from_tag, from_entity = parse_label(from_label)
    to_tag, to_entity = parse_label(to_label)
    if from_tag == 'O' and to_tag in 'IO':
        return True
    if from_tag == 'B' and to_tag == 'I':
        return from_entity == to_entity
    if from_tag == 'B' and to_tag in 'BO':
        return True
    if from_tag == 'I' and to_tag == 'I':
        return from_entity == to_entity
    if from_tag == 'I' and to_tag in 'BO':
        return True
    return False

def iob2_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    if from_label is None:
        return to_label is not None and to_label[0] in 'BO'
    if to_label is None:
        return True
    from_tag, from_entity = parse_label(from_label)
    to_tag, to_entity = parse_label(to_label)
    if from_tag == 'O' and to_tag in 'BO':
        return True
    if from_tag == 'B' and to_tag == 'I':
        return from_entity == to_entity
    if from_tag == 'B' and to_tag in 'BO':
        return True
    if from_tag == 'I' and to_tag == 'I':
        return from_entity == to_entity
    if from_tag == 'I' and to_tag in 'BO':
        return True
    return False

def bilou_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    if from_label is None:
        return to_label is not None and to_label[0] in 'BUO'
    from_tag, from_entity = parse_label(from_label)
    if to_label is None:
        return from_tag in 'LUO'
    to_tag, to_entity = parse_label(to_label)
    if from_tag == 'O' and to_tag in 'BUO':
        return True
    if from_tag == 'B' and to_tag in 'IL':
        return from_entity == to_entity
    if from_tag == 'I' and to_tag in 'IL':
        return from_entity == to_entity
    if from_tag == 'L' and to_tag in 'BUO':
        return True
    if from_tag == 'U' and to_tag in 'BUO':
        return True
    return False

def iobes_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    if from_label is None:
        return to_label is not None and to_label[0] in 'OBS'
    from_tag, from_entity = parse_label(from_label)
    if to_label is None:
        return from_tag in 'EUO'
    to_tag, to_entity = parse_label(to_label)
    if from_tag == 'O' and to_tag in 'OBS':
        return True
    if from_tag == 'B' and to_tag in 'IE':
        return from_entity == to_entity
    if from_tag == 'I' and to_tag in 'IE':
        return from_entity == to_entity
    if from_tag == 'E' and to_tag in 'OBS':
        return True
    if from_tag == 'S' and to_tag in 'OBS':
        return True
    return False



def raw_to_spans(labels: list[str]) -> list[TokenSpan]:
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not raw_valid_label(label):
            raise ValueError(f'invalid {label=}')
        if not raw_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'invalid transition {prev!r} -> {label!r}')
        tag, entity = parse_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag is None:
            if entity != span_entity:
                spans.append(TokenSpan(start, i, span_entity))
                start = i
                span_entity = entity
        previous_label = label
    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))
    return spans

def io_to_spans(labels: list[str]) -> list[TokenSpan]:
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not io_valid_label(label):
            raise ValueError(f'invalid {label=}')
        if not io_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'invalid transition {prev!r} -> {label!r}')
        tag, entity = parse_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'I':
            if entity != span_entity:
                spans.append(TokenSpan(start, i, span_entity))
                start = i
                span_entity = entity
        previous_label = label
    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))
    return spans

def iob1_to_spans(labels: list[str]) -> list[TokenSpan]:
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not iob1_valid_label(label):
            raise ValueError(f'invalid {label=}')
        if not iob1_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'invalid transition {prev!r} -> {label!r}')
        tag, entity = parse_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'B':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
            start = i
            span_entity = entity
        elif tag == 'I':
            if start is None:
                start = i
                span_entity = entity
        previous_label = label
    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))
    return spans

def iob2_to_spans(labels: list[str]) -> list[TokenSpan]:
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not iob2_valid_label(label):
            raise ValueError(f'invalid {label=}')
        if not iob2_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'invalid transition {prev!r} -> {label!r}')
        tag, entity = parse_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'B':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
            start = i
            span_entity = entity
        previous_label = label
    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))
    return spans

def bilou_to_spans(labels: list[str]) -> list[TokenSpan]:
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not bilou_valid_label(label):
            raise ValueError(f'invalid {label=}')
        if not bilou_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'invalid transition {prev!r} -> {label!r}')
        tag, entity = parse_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'B':
            start = i
            span_entity = entity
        elif tag == 'L':
            spans.append(TokenSpan(start, i, span_entity))
            start = span_entity = None
        elif tag == 'U':
            spans.append(TokenSpan(i, i+1, span_entity))
        previous_label = label
    if not bilou_valid_transition(labels[-1], None):
        raise ValueError(f'invalid transition {label!r} -> END')
    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))
    return spans

def iobes_to_spans(labels: list[str]) -> list[TokenSpan]:
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not iobes_valid_label(label):
            raise ValueError(f'invalid {label=}')
        if not iobes_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'invalid transition {prev!r} -> {label!r}')
        tag, entity = parse_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'B':
            start = i
            span_entity = entity
        elif tag == 'E':
            spans.append(TokenSpan(start, i, span_entity))
            start = span_entity = None
        elif tag == 'S':
            spans.append(TokenSpan(i, i+1, span_entity))
        previous_label = label
    if not iobes_valid_transition(labels[-1], None):
        raise ValueError(f'invalid transition {label!r} -> END')
    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))
    return spans



def spans_to_raw(num_tokens: int, spans: list[TokenSpan]) -> list[str]:
    if Span.any_overlap(spans):
        raise ValueError('overlapping spans')
    labels = ['O'] * num_tokens
    for span in spans:
        if span.label is None:
            raise ValueError('label missing')
        for i in span.range():
            labels[i] = span.label
    return labels

def spans_to_io(num_tokens: int, spans: list[TokenSpan]) -> list[str]:
    if Span.any_overlap(spans):
        raise ValueError('overlapping spans')
    labels = ['O'] * num_tokens
    for span in spans:
        for i in span.range():
            labels[i] = f'I-{span.label}' if span.label else 'I'
    return labels

def spans_to_iob1(num_tokens: int, spans: list[TokenSpan]) -> list[str]:
    if Span.any_overlap(spans):
        raise ValueError('overlapping spans')
    labels = ['O'] * num_tokens
    for span in spans:
        for i in span.range():
            if i == span.first and i > 0 and labels[i-1] != 'O':
                label = f'B-{span.label}' if span.label else 'B'
            else:
                label = f'I-{span.label}' if span.label else 'I'
            labels[i] = label
    return labels

def spans_to_iob2(num_tokens: int, spans: list[TokenSpan]) -> list[str]:
    if Span.any_overlap(spans):
        raise ValueError('overlapping spans')
    labels = ['O'] * num_tokens
    for span in spans:
        for i in span.range():
            if i == span.first:
                label = f'B-{span.label}' if span.label else 'B'
            else:
                label = f'I-{span.label}' if span.label else 'I'
            labels[i] = label
    return labels

def spans_to_bilou(num_tokens: int, spans: list[TokenSpan]) -> list[str]:
    if Span.any_overlap(spans):
        raise ValueError('overlapping spans')
    labels = ['O'] * num_tokens
    for span in spans:
        for i in span.range():
            if span.length() == 1:
                label = f'U-{span.label}' if span.label else 'U'
            elif i == span.first:
                label = f'B-{span.label}' if span.label else 'B'
            elif i == span.last:
                label = f'L-{span.label}' if span.label else 'L'
            else:
                label = f'I-{span.label}' if span.label else 'I'
            labels[i] = label
    return labels

def spans_to_iobes(num_tokens: int, spans: list[TokenSpan]) -> list[str]:
    if Span.any_overlap(spans):
        raise ValueError('overlapping spans')
    labels = ['O'] * num_tokens
    for span in spans:
        for i in span.range():
            if span.length() == 1:
                label = f'S-{span.label}' if span.label else 'S'
            elif i == span.first:
                label = f'B-{span.label}' if span.label else 'B'
            elif i == span.last:
                label = f'E-{span.label}' if span.label else 'E'
            else:
                label = f'I-{span.label}' if span.label else 'I'
            labels[i] = label
    return labels



def io_to_iob1(labels: list[str]) -> list[str]:
    return spans_to_iob1(len(labels), io_to_spans(labels))

def io_to_iob2(labels: list[str]) -> list[str]:
    return spans_to_iob2(len(labels), io_to_spans(labels))

def io_to_bilou(labels: list[str]) -> list[str]:
    return spans_to_bilou(len(labels), io_to_spans(labels))

def io_to_iobes(labels: list[str]) -> list[str]:
    return spans_to_iobes(len(labels), io_to_spans(labels))

def io_to_raw(labels: list[str]) -> list[str]:
    return spans_to_raw(len(labels), io_to_spans(labels))

def iob1_to_io(labels: list[str]) -> list[str]:
    return spans_to_io(len(labels), iob1_to_spans(labels))

def iob1_to_iob2(labels: list[str]) -> list[str]:
    return spans_to_iob2(len(labels), iob1_to_spans(labels))

def iob1_to_bilou(labels: list[str]) -> list[str]:
    return spans_to_bilou(len(labels), iob1_to_spans(labels))

def iob1_to_iobes(labels: list[str]) -> list[str]:
    return spans_to_iobes(len(labels), iob1_to_spans(labels))

def iob1_to_raw(labels: list[str]) -> list[str]:
    return spans_to_raw(len(labels), iob1_to_spans(labels))

def iob2_to_io(labels: list[str]) -> list[str]:
    return spans_to_io(len(labels), iob2_to_spans(labels))

def iob2_to_iob1(labels: list[str]) -> list[str]:
    return spans_to_iob1(len(labels), iob2_to_spans(labels))

def iob2_to_bilou(labels: list[str]) -> list[str]:
    return spans_to_bilou(len(labels), iob2_to_spans(labels))

def iob2_to_iobes(labels: list[str]) -> list[str]:
    return spans_to_iobes(len(labels), iob2_to_spans(labels))

def iob2_to_raw(labels: list[str]) -> list[str]:
    return spans_to_raw(len(labels), iob2_to_spans(labels))

def bilou_to_io(labels: list[str]) -> list[str]:
    return spans_to_io(len(labels), bilou_to_spans(labels))

def bilou_to_iob1(labels: list[str]) -> list[str]:
    return spans_to_iob1(len(labels), bilou_to_spans(labels))

def bilou_to_iob2(labels: list[str]) -> list[str]:
    return spans_to_iob2(len(labels), bilou_to_spans(labels))

def bilou_to_iobes(labels: list[str]) -> list[str]:
    return spans_to_iobes(len(labels), bilou_to_spans(labels))

def bilou_to_raw(labels: list[str]) -> list[str]:
    return spans_to_raw(len(labels), bilou_to_spans(labels))

def iobes_to_io(labels: list[str]) -> list[str]:
    return spans_to_io(len(labels), iobes_to_spans(labels))

def iobes_to_iob1(labels: list[str]) -> list[str]:
    return spans_to_iob1(len(labels), iobes_to_spans(labels))

def iobes_to_iob2(labels: list[str]) -> list[str]:
    return spans_to_iob2(len(labels), iobes_to_spans(labels))

def iobes_to_bilou(labels: list[str]) -> list[str]:
    return spans_to_bilou(len(labels), iobes_to_spans(labels))

def iobes_to_raw(labels: list[str]) -> list[str]:
    return spans_to_raw(len(labels), iobes_to_spans(labels))

def raw_to_io(labels: list[str]) -> list[str]:
    return spans_to_io(len(labels), raw_to_spans(labels))

def raw_to_iob1(labels: list[str]) -> list[str]:
    return spans_to_iob1(len(labels), raw_to_spans(labels))

def raw_to_iob2(labels: list[str]) -> list[str]:
    return spans_to_iob2(len(labels), raw_to_spans(labels))

def raw_to_bilou(labels: list[str]) -> list[str]:
    return spans_to_bilou(len(labels), raw_to_spans(labels))

def raw_to_iobes(labels: list[str]) -> list[str]:
    return spans_to_iobes(len(labels), raw_to_spans(labels))

