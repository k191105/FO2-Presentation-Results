from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union, get_args

@dataclass(frozen=True)
class Variable:
    """A variable in a formula, restricted to 'x' or 'y'."""
    name: Literal['x', 'y']

    def __str__(self) -> str:
        return self.name

x = Variable('x')
y = Variable('y')

@dataclass(frozen=True)
class Predicate:
    """A predicate symbol with a specific arity."""
    symbol: str
    arity: int

@dataclass(frozen=True)
class Atom:
    """An atomic formula like P(x, y)."""
    predicate: Predicate
    terms: tuple[Variable, ...]

    def __post_init__(self):
        if self.predicate.arity != len(self.terms):
            raise ValueError(f"Arity mismatch for {self.predicate.symbol}")

    def __str__(self) -> str:
        return f"{self.predicate.symbol}({', '.join(map(str, self.terms))})"

@dataclass(frozen=True)
class Not:
    """A negated formula."""
    formula: Formula

    def __str__(self) -> str:
        return f"¬({self.formula})"

@dataclass(frozen=True)
class And:
    """A conjunction of two formulas."""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

@dataclass(frozen=True)
class Or:
    """A disjunction of two formulas."""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

@dataclass(frozen=True)
class Implies:
    """An implication between two formulas."""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"

@dataclass(frozen=True)
class Exists:
    """An existentially quantified formula."""
    variable: Variable
    formula: Formula

    def __str__(self) -> str:
        return f"∃{self.variable}.({self.formula})"

@dataclass(frozen=True)
class ForAll:
    """A universally quantified formula."""
    variable: Variable
    formula: Formula

    def __str__(self) -> str:
        return f"∀{self.variable}.({self.formula})"

Formula = Union[Atom, Not, And, Or, Implies, Exists, ForAll]

def is_atom(node: Formula) -> bool: return isinstance(node, Atom)
def is_not(node: Formula) -> bool: return isinstance(node, Not)
def is_and(node: Formula) -> bool: return isinstance(node, And)
def is_or(node: Formula) -> bool: return isinstance(node, Or)
def is_implies(node: Formula) -> bool: return isinstance(node, Implies)
def is_exists(node: Formula) -> bool: return isinstance(node, Exists)
def is_forall(node: Formula) -> bool: return isinstance(node, ForAll)

def free_vars(node: Formula) -> set[str]:
    """Computes the set of free variables in a formula."""
    if is_atom(node):
        return {term.name for term in node.terms}
    if is_not(node):
        return free_vars(node.formula)
    if is_and(node) or is_or(node) or is_implies(node):
        return free_vars(node.left) | free_vars(node.right)
    if is_exists(node) or is_forall(node):
        return free_vars(node.formula) - {node.variable.name}
    raise TypeError(f"Unknown formula type: {type(node)}")

def get_predicates(node: Formula) -> set[Predicate]:
    """Computes the set of predicates in a formula."""
    if is_atom(node):
        return {node.predicate}
    if is_not(node):
        return get_predicates(node.formula)
    if is_and(node) or is_or(node) or is_implies(node):
        return get_predicates(node.left) | get_predicates(node.right)
    if is_exists(node) or is_forall(node):
        return get_predicates(node.formula)
    raise TypeError(f"Unknown formula type: {type(node)}")
