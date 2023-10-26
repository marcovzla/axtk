from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
import regex
import torch
from axtk.generation_utils.logits_processors.grammar_parser import GrammarParser, Grammar, Symbol, Terminal, Production, epsilon, end_of_input
from axtk.generation_utils.logits_processors.acceptable_logits_processor import AcceptableLogitsProcessor



class GrammarLogitsProcessor(AcceptableLogitsProcessor):
    def __init__(
            self,
            grammar: str | Grammar,
            stop_regex: str | list[str],
            **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        stop_regex.append(regex.escape(self.tokenizer.eos_token))

        self.stop_pattern = regex.compile('|'.join(f'(?:{rx})' for rx in set(stop_regex)))
        self.grammar = grammar if isinstance(grammar, Grammar) else GrammarParser().parse(grammar)
        self.state = ParserState(self.grammar, self.stop_pattern)

    def update_state(self, input_ids: torch.Tensor):
        self.simulate_parse(input_ids[0][self.prefix_length:])
        self.prefix_length = input_ids.size(1)

    def is_acceptable(self, proposed_token_id: int) -> bool:
        proposed_token = self.tokenizer.decode(proposed_token_id)
        return self.state.is_acceptable(proposed_token)

    def simulate_parse(self, new_ids: list[int]):
        for token_id in new_ids:
            text = self.tokenizer.decode(token_id)
            self.state.accept(text)
            if self.state.failure():
                raise ValueError('Invalid token sequence')



class ParserState:
    def __init__(self, grammar: Grammar, stop_pattern: regex.Pattern[str]):
        self.grammar = grammar
        self.stop_pattern = stop_pattern
        self.parsed = ''
        self.stacks = [
            Stack(self.grammar, self.stop_pattern).push(production)
            for production in self.grammar.start_productions()
        ]

    def success(self) -> bool:
        return any(s.success for s in self.stacks)

    def failure(self) -> bool:
        return len(self.stacks) == 0

    def is_acceptable(self, text: str) -> bool:
        stacks = self.propose(text)
        return len(stacks) > 0

    def accept(self, text: str):
        self.stacks = self.propose(text)
        self.parsed += text

    def propose(self, text: str) -> list[Stack]:
        return [
            new_stack
            for stack in self.stacks
            for new_stack in stack.match_input(text, self.parsed)
        ]



class Stack:
    def __init__(
            self,
            grammar: Grammar,
            stop_pattern: regex.Pattern[str],
            frames: Optional[list[StackFrame]] = None,
            consumed_chars: int = 0,
    ):
        self.grammar = grammar
        self.stop_pattern = stop_pattern
        self.frames = frames or []
        self.consumed_chars = consumed_chars
        self.success = False

    @property
    def top(self) -> StackFrame:
        return self.frames[-1]

    def copy(self):
        return Stack(
            grammar=self.grammar,
            stop_pattern=self.stop_pattern,
            frames=deepcopy(self.frames),
            consumed_chars=self.consumed_chars,
        )

    def is_empty(self):
        return len(self.frames) == 0

    def push(self, production: Production):
        self.frames.append(StackFrame(production))
        return self

    def pop(self):
        return self.frames.pop()

    def next_terminals(self) -> set[Symbol]:
        if self.is_empty():
            return {end_of_input}
        else:
            return self.grammar.FIRST[self.top.symbol]

    def consume_text(self, text: str):
        # accumulate number of consumed chars
        self.consumed_chars += len(text)
        # consume the next grammar symbol, popping from the stack as needed
        while not self.is_empty():
            self.top.consume_symbol()
            if self.top.is_done():
                self.pop()
            else:
                break
        return self

    def match_input(self, proposed: str, previous: str) -> list[Stack]:
        # if an empty string was proposed, then there is nothing to do
        if proposed == '':
            return [self]
        # get portion of the previously generated text that hasn't been consumed by the grammar
        unconsumed = previous[self.consumed_chars:]
        if self.is_empty():
            if self.success:
                # if parser is in a successful state, any extra input invalidates it
                return []
            # The full input has been parsed successfully so far.
            # Check if the LM produces a stop pattern.
            if m := self.stop_pattern.fullmatch(unconsumed + proposed, partial=True):
                if not m.partial:
                    # stop pattern matched fully
                    # self.consume_text(unconsumed + proposed)
                    stack = self.copy()
                    stack.success = True
                return [stack]
            else:
                return []
        elif self.top.symbol.is_nonterminal():
            # If next symbol is a non-terminal, then we need to push the corresponding rule to the top of the stack.
            # If there are several rules with the same name, then one new stack must be created for each rule.
            return [
                stack
                for production in self.grammar[self.top.symbol]
                for stack in self.copy().push(production).match_input(proposed, previous)
            ]
        elif self.top.symbol.is_terminal():
            # check that unconsumed text matches
            if unconsumed_match := self.top.fullmatch(unconsumed):
                # if proposed text still matches then keep going
                if self.top.fullmatch(unconsumed + proposed):
                    return [self]
                # if proposed text doesn't match but unconsumed text matched fully,
                # then consume unconsumed text and try to match proposed text by itself
                elif not unconsumed_match.partial:
                    return self.copy().consume_text(unconsumed).match_input(proposed, previous)
            # failed to match
            return []
        elif self.top.symbol.is_epsilon():
            return self.copy().consume_text('').match_input(proposed, previous)
        else:
            raise Exception(f'unexpected symbol: {self.top.symbol}')



@dataclass
class StackFrame:
    production: Production
    pointer: int = 0

    @property
    def symbol(self):
        return self.production.rhs[self.pointer]

    def consume_symbol(self):
        while True:
            self.pointer += 1
            if self.is_done() or self.symbol != epsilon:
                break

    def is_done(self) -> bool:
        return self.pointer >= len(self.production.rhs)

    def fullmatch(self, token: str) -> Optional[regex.Match[str]]:
        if isinstance(self.symbol, Terminal):
            return self.symbol.fullmatch(token)
