from formula import *
from typing import Any, Dict, Set

Assignment = Dict[str, int]

class Model:
    """A finite model, which interprets predicate symbols over a domain."""

    def __init__(self, domain: set[int], interpretation: dict[str, set[tuple[int, ...]]]):
        self.domain = domain
        self.interpretation = interpretation

    def eval(self, formula: Formula, assignment: Assignment | None = None) -> bool:
        """
        Evaluates a formula in the model with a given variable assignment.
        """
        assignment = assignment or {}
        # First, check for unassigned free variables.
        unassigned_vars = free_vars(formula) - assignment.keys()
        if unassigned_vars:
            raise ValueError(f"Unassigned free variables: {unassigned_vars}")

        if is_atom(formula):
            # Unknown predicate
            if formula.predicate.symbol not in self.interpretation:
                raise ValueError(f"Unknown predicate symbol: {formula.predicate.symbol}")
            
            interpretation_set = self.interpretation[formula.predicate.symbol]
            
            # Arity mismatch
            # We can get arity from one of the tuples in the interpretation if it's not empty.
            if interpretation_set:
                some_tuple = next(iter(interpretation_set))
                if len(some_tuple) != formula.predicate.arity:
                    raise ValueError(f"Arity mismatch for {formula.predicate.symbol}")

            # Evaluate atom
            term_values = tuple(assignment[term.name] for term in formula.terms)
            return term_values in interpretation_set

        if is_not(formula):
            return not self.eval(formula.formula, assignment)

        if is_and(formula):
            return self.eval(formula.left, assignment) and self.eval(formula.right, assignment)

        if is_or(formula):
            return self.eval(formula.left, assignment) or self.eval(formula.right, assignment)

        if is_implies(formula):
            return not self.eval(formula.left, assignment) or self.eval(formula.right, assignment)

        if is_exists(formula):
            var_name = formula.variable.name
            for domain_element in self.domain:
                new_assignment = assignment.copy()
                new_assignment[var_name] = domain_element
                if self.eval(formula.formula, new_assignment):
                    return True
            return False

        if is_forall(formula):
            var_name = formula.variable.name
            for domain_element in self.domain:
                new_assignment = assignment.copy()
                new_assignment[var_name] = domain_element
                if not self.eval(formula.formula, new_assignment):
                    return False
            return True
        
        raise TypeError(f"Unknown formula type: {type(formula)}")
