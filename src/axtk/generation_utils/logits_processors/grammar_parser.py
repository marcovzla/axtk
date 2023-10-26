from __future__ import annotations
from ast import literal_eval
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional
import regex
from lark import Lark, Transformer, v_args, Discard
from axtk.regex_utils import delimited_string, get_pattern_from_regex_literal


# special symbols
# ---------------

EPSILON = 'ε'

END_OF_INPUT = '$'


# operators
# ---------

IS_DEFINED_AS = ':'

DISJUNCTION = '|'

ZERO_OR_MORE = '*'

ONE_OR_MORE = '+'

ZERO_OR_ONE = '?'

OPEN_PARENS = '('

CLOSE_PARENS = ')'


# whitespace
# ----------

NEWLINE = r'\n'

COMMENT = r'#.*'

WHITESPACE = rf'(?:\s+|{COMMENT})+'

WHITESPACE_INLINE = rf'[^\S\f\r\n]+(?:{COMMENT})?'


# literals
# --------

IDENTIFIER = r'[^\W\d]\w*'

STRING = delimited_string('\'"', return_string=True)

# regexes are delimited by forward-slashes in lark grammars,
# so forward slashes in the pattern must be escaped
REGEX = delimited_string('/', return_string=True).replace('/', r'\/')


# grammar for parsing bnf grammars
# --------------------------------

GRAMMAR_START_SYMBOL = 'start'

GRAMMAR = rf"""

    {GRAMMAR_START_SYMBOL}: production+

    production: WS? IDENTIFIER WS? "{IS_DEFINED_AS}" WS? disjunction WS_INLINE? "{NEWLINE}" WS?

    disjunction: (concatenation WS? "{DISJUNCTION}" WS?)* concatenation

    concatenation: (quant_atom WS)* quant_atom

    ?quant_atom: zero_or_one
               | zero_or_more
               | one_or_more
               | atom

    zero_or_one: atom WS? "{ZERO_OR_ONE}"

    zero_or_more: atom WS? "{ZERO_OR_MORE}"

    one_or_more: atom WS? "{ONE_OR_MORE}"

    ?atom: group
         | IDENTIFIER
         | STRING
         | REGEX
         | EPSILON

    ?group: "{OPEN_PARENS}" WS? disjunction WS? "{CLOSE_PARENS}"

    IDENTIFIER: /{IDENTIFIER}/

    STRING: /{STRING}/

    REGEX: /{REGEX}/

    EPSILON: "{EPSILON}"

    WS: /{WHITESPACE}/

    WS_INLINE: /{WHITESPACE_INLINE}/

"""


# target grammars defaults
# ------------------------

START_SYMBOL = 'start'

IGNORE_PATTERN = r'[^\S\f\r\n]+'



class GrammarParser:
    """A parser for processing grammars."""

    def __init__(
            self,
            grammar: str = GRAMMAR,
            grammar_start_symbol: str = GRAMMAR_START_SYMBOL,
            start_symbol: str = START_SYMBOL,
            ignore_pattern: Optional[str] = IGNORE_PATTERN,
    ):
        self.lark = Lark(grammar, start=grammar_start_symbol)
        self.transformer = GrammarTransformer(start_symbol, ignore_pattern)

    def parse(self, text: str) -> Grammar:
        tree = self.lark.parse(text)
        grammar = self.transformer.transform(tree)
        return grammar



class GrammarTransformer(Transformer):
    def __init__(self, start_symbol: str, ignore_pattern: Optional[str]):
        super().__init__()
        self.start_symbol = start_symbol
        self.ignore_pattern = ignore_pattern
        self.reset()

    def reset(self):
        """
        Resets the state of the transformer to its initial configuration.

        This method reinitializes the transformer's state by resetting the counter 
        `self.new_rules_counter` to 0 and clearing the list `self.auxiliary_productions`.
        This ensures that the transformer is in a clean state, ready for a new round of 
        processing, without any carry-over from previous operations.
        """
        self.new_rules_counter = 0
        self.auxiliary_productions = []

    def start(self, productions) -> Grammar:
        productions = self.prepare_productions(productions)
        return Grammar(productions, self.start_symbol)

    @v_args(inline=True)
    def production(self, lhs, rhs) -> tuple[NonTerminal, AstNode]:
        return (lhs, rhs)

    def disjunction(self, nodes) -> AstNode:
        return nodes[0] if len(nodes) == 1 else Disjunction(nodes)

    def concatenation(self, nodes) -> AstNode:
        return nodes[0] if len(nodes) == 1 else Concatenation(nodes)

    @v_args(inline=True)
    def zero_or_one(self, node) -> ZeroOrOne:
        return ZeroOrOne(node)

    @v_args(inline=True)
    def zero_or_more(self, node) -> ZeroOrMore:
        return ZeroOrMore(node)

    @v_args(inline=True)
    def one_or_more(self, node) -> OneOrMore:
        return OneOrMore(node)

    def IDENTIFIER(self, token) -> NonTerminal:
        return NonTerminal(str(token))

    def STRING(self, token) -> String:
        return String(token, self.ignore_pattern)

    def REGEX(self, token) -> Regex:
        return Regex(token, self.ignore_pattern)

    def EPSILON(self, token) -> Epsilon:
        return epsilon

    def WS(self, token):
        return Discard

    def WS_INLINE(self, token):
        return Discard

    def prepare_productions(self, productions: Iterable[tuple[NonTerminal, AstNode]]) -> list[Production]:
        """
        Simplifies the given productions by removing auxiliary parsing constructs.

        This method processes the input productions to eliminate auxiliary parsing
        constructs such as Concatenation, Disjunction, ZeroOrOne, ZeroOrMore, and
        OneOrMore, simplifying the grammar. The simplification is carried out per production,
        and any auxiliary productions generated during this process are stored in
        `self.auxiliary_productions`.

        The method begins by resetting the transformer's state through a call to `self.reset()`.
        It then iterates through the given productions, simplifying each one using the
        `simplify_node` method. The simplified productions, along with any auxiliary
        productions generated during the process, are collected and returned as a list
        of `Production` instances.

        Args:
            productions (Iterable[tuple[NonTerminal, AstNode]]): A collection of productions
                to be simplified. Each production is represented as a tuple where the first
                element is a NonTerminal and the second element is a Node.

        Returns:
            list[Production]: A list of simplified productions, each represented as a
                `Production` instance.
        """
        # reset the transformer's state
        self.reset()
        # simplify productions
        simple_productions = [
            Production(lhs, simple_rhs)
            for lhs, rhs in productions
            for simple_rhs in self.simplify_node(rhs)
        ]
        # append all auxiliary productions generated during simplification
        simple_productions += self.auxiliary_productions
        # return all simple productions
        return simple_productions

    def new_non_terminal(self) -> NonTerminal:
        """
        Generates and returns a unique NonTerminal for naming auxiliary productions.

        This method creates a new NonTerminal instance with a unique identifier, which
        is suitable for naming auxiliary productions. The unique identifier is based
        on a counter (`self.new_rules_counter`) that increments with each call to ensure
        uniqueness.

        The counter `self.new_rules_counter` is incremented by 1 each time this method
        is called, to ensure the uniqueness of the NonTerminal identifiers.

        Returns:
            NonTerminal: A new NonTerminal instance with a unique identifier.
        """
        non_terminal = NonTerminal(f'non_terminal_{self.new_rules_counter}')
        self.new_rules_counter += 1
        return non_terminal

    def simplify_node(self, node: AstNode) -> list[list[Symbol]]:
        """
        Simplify the given node into a disjunction of concatenations.

        This method takes a node, applies a set of simplification rules to it,
        and returns a list of lists of symbols. Each inner list represents a
        concatenation of symbols, and the outer list represents a disjunction of
        these concatenations. Any auxiliary productions generated during this
        process will be stored in self.auxiliary_productions.

        Args:
            node (AstNode): The node to be simplified.

        Returns:
            list[list[Symbol]]: A list of lists of symbols representing a disjunction
                of concatenations.

        Examples:
            Consider a node representing the regular expression (a|b)c. After
            simplification, this method would return [[Symbol('a'), Symbol('c')],
            [Symbol('b'), Symbol('c')]].
        """
        if isinstance(node, Symbol):
            return [[node]]
        elif isinstance(node, Disjunction):
            return [
                production
                for n in node.nodes
                for production in self.simplify_node(n)
            ]
        elif isinstance(node, Concatenation):
            productions = self.simplify_node(node.nodes[0])
            for n in node.nodes[1:]:
                productions = [
                    p1 + p2
                    for p1 in productions
                    for p2 in self.simplify_node(n)
                ]
            return productions
        elif isinstance(node, ZeroOrMore):
            lhs = self.new_non_terminal()
            for rhs in self.simplify_node(node.node):
                self.auxiliary_productions.append(Production(lhs, rhs + [lhs]))
            self.auxiliary_productions.append(Production(lhs, [epsilon]))
            return [[lhs]]
        elif isinstance(node, OneOrMore):
            lhs = self.new_non_terminal()
            for rhs in self.simplify_node(node.node):
                self.auxiliary_productions.append(Production(lhs, rhs + [lhs]))
            return [[lhs]]
        elif isinstance(node, ZeroOrOne):
            lhs = self.new_non_terminal()
            for rhs in self.simplify_node(node.node):
                self.auxiliary_productions.append(Production(lhs, rhs))
            self.auxiliary_productions.append(Production(lhs, [epsilon]))
            return [[lhs]]
        else:
            raise ValueError(f'unsupported node: {node!r}')



class Symbol:
    def __repr__(self):
        return f'<{self.__class__.__name__}: {self}>'

    def is_terminal(self) -> bool:
        return False

    def is_nonterminal(self) -> bool:
        return False

    def is_epsilon(self) -> bool:
        return False

    def is_end_of_input(self) -> bool:
        return False



class Terminal(Symbol):
    def __init__(self, token: str, ignore: Optional[str], escape: bool):
        self.token = token
        self.ignore = ignore
        self.escape = escape
        self.value = self.parse_token(self.token)
        self.pattern = self.compile_pattern(self.value)

    def __str__(self):
        return self.token

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def is_terminal(self) -> bool:
        return True

    def parse_token(self, token: str) -> str:
        raise NotImplementedError()

    def compile_pattern(self, pattern: str) -> regex.Pattern[str]:
        if self.escape:
            pattern = regex.escape(pattern)
        if self.ignore:
            pattern = f'(?:{self.ignore})?{pattern}'
        return regex.compile(pattern)

    def fullmatch(self, string: str) -> Optional[regex.Match[str]]:
        return self.pattern.fullmatch(string, partial=True)



class String(Terminal):
    def __init__(self, token: str, ignore: Optional[str] = None):
        super().__init__(token, ignore, escape=True)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    # __hash__ must be set explicitly to the parent's implementation
    # because __eq__ was overridden
    __hash__ = Terminal.__hash__

    def parse_token(self, token: str) -> str:
        return literal_eval(token)



class Regex(Terminal):
    def __init__(self, token: str, ignore: Optional[str] = None):
        super().__init__(token, ignore, escape=False)

    def parse_token(self, token: str) -> str:
        return get_pattern_from_regex_literal(token)



class NonTerminal(Symbol):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, NonTerminal):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False

    def __hash__(self):
        return hash(self.name)

    def is_nonterminal(self) -> bool:
        return True



class Epsilon(Symbol):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self):
        return EPSILON

    def is_epsilon(self) -> bool:
        return True



class EndOfInput(Symbol):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self):
        return END_OF_INPUT

    def is_end_of_input(self) -> bool:
        return True



# classes that are useful during parsing,
# not part of the final grammar

class Concatenation:
    def __init__(self, nodes: list[AstNode]):
        self.nodes = nodes

class Disjunction:
    def __init__(self, nodes: list[AstNode]):
        self.nodes = nodes

class ZeroOrOne:
    def __init__(self, node: AstNode):
        self.node = node

class ZeroOrMore:
    def __init__(self, node: AstNode):
        self.node = node

class OneOrMore:
    def __init__(self, node: AstNode):
        self.node = node



# singletons
epsilon = Epsilon()
end_of_input = EndOfInput()

# type alias
AstNode = Symbol | Concatenation | Disjunction | ZeroOrOne | ZeroOrMore | OneOrMore



class Production:
    def __init__(self, lhs: NonTerminal, rhs: list[Symbol]):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        rhs = ' '.join(map(str, self.rhs))
        return f'{self.lhs} {IS_DEFINED_AS} {rhs}'



class Grammar:
    def __init__(self, productions: list[Production], start_symbol: str):
        self.productions = productions
        self.start_symbol = start_symbol
        self.FIRST = self.compute_first_sets()

    def __str__(self):
        return '\n\n'.join(map(str, self.productions))

    def __iter__(self):
        return iter(self.non_terminals())

    def __getitem__(self, key):
        return self.get_productions(key)

    def start_productions(self) -> list[Production]:
        return self.get_productions(self.start_symbol)

    def get_productions(self, lhs):
        return [
            production
            for production in self.productions
            if production.lhs == lhs
        ]

    def non_terminals(self) -> set[NonTerminal]:
        return set(production.lhs for production in self.productions)

    def next_terminals(self, obj: Symbol | str | list[Symbol] | Production) -> set[Terminal]:
        if isinstance(obj, str):
            symbol = NonTerminal(obj)
        elif isinstance(obj, list):
            symbol = obj[0]
        elif isinstance(obj, Production):
            symbol = obj.rhs[0]
        else:
            symbol = obj
        if symbol not in self.FIRST:
            raise ValueError(f"Symbol '{symbol}' not in FIRST set.")
        return self.FIRST[symbol]

    def compute_first_sets(self) -> dict[Symbol, set[Terminal]]:
        """Compute FIRST sets for grammar."""

        FIRST = defaultdict(set)

        # Initialize FIRST set for terminals
        for production in self.productions:
            for symbol in production.rhs:
                if symbol.is_terminal():
                    FIRST[symbol] = {symbol}

        changed = True
        while changed:
            changed = False
            for production in self.productions:
                temp_first = set()
                can_produce_epsilon = True

                for symbol in production.rhs:
                    if symbol.is_nonterminal():
                        temp_first |= (FIRST[symbol] - {epsilon})
                    else:  # symbol is a terminal
                        temp_first.add(symbol)

                    if epsilon not in FIRST[symbol]:
                        can_produce_epsilon = False
                        break

                if can_produce_epsilon:
                    temp_first.add(epsilon)

                if not FIRST[production.lhs].issuperset(temp_first):
                    FIRST[production.lhs] |= temp_first
                    changed = True

        return dict(FIRST)
