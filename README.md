# FO² Model Theory Presentation Files

This project contains implementations for a satisfiability checker and model explorer for the binary fragment of First-Order Logic(FO²), where formulas are restricted to at most two distinct variables. The code in this project was usedf to deliver a presentation on the decideability of FO2 to the Australasian Seminar in Logic. For a nicer exposition of the ideas implemented here, ind the presentation here: https://docs.google.com/presentation/d/1SAHjaGzuCCPHUVZiZ7IF5opHbTyGHZb0F1T9v4tCk3E/edit?slide=id.g36f1f4ad801_0_11988#slide=id.g36f1f4ad801_0_11988



## File Overview

Files overview generated with Claude in Cursor:

### `formula.py`
**Goal**: Define the abstract syntax tree for FO² formulas.

**Implementation**: Uses Python dataclasses to represent variables (restricted to 'x' and 'y'), predicates (with arity), atoms, logical connectives (And, Or, Not, Implies), and quantifiers (Exists, ForAll). Includes helper functions to compute free variables and extract predicates from formulas.

**Outcome**: Provides a type-safe, compositional representation of FO² formulas that can be programmatically constructed and analyzed.

### `model.py`
**Goal**: Implement semantic evaluation of FO² formulas over finite models.

**Implementation**: The Model class represents a finite structure with a domain (set of integers) and an interpretation (mapping predicate symbols to sets of tuples). The `eval` method recursively evaluates formulas by handling atoms (lookup in interpretation), logical connectives (boolean operations), and quantifiers (iteration over domain elements with variable assignments).

**Outcome**: Correctly determines whether a given formula is satisfied by a specific finite model, handling variable binding and quantifier scope properly.

### `generate_models.py`
**Goal**: Generate finite models either exhaustively or randomly.

**Implementation**: The `enumerate_models` function exhaustively generates all possible models for a domain size by iterating through all graph structures (for binary predicates) and all colorings (for unary predicates). The `sample_model` function generates random models by sampling edges and vertex properties according to configured probabilities. The `count_models` function calculates the total number of labeled models as 2^(n choose 2) × 2^(k×n).

**Outcome**: Enables both complete enumeration for small domains (n ≤ 5) and efficient random sampling for large domains (n up to 100), providing the data for satisfiability analysis.

### `experiments.py`
**Goal**: Run experiments to measure satisfiability rates and find satisfying models.

**Implementation**: The `run_exhaustive` function tests all models for small domains, using Weisfeiler-Lehman graph hashing to filter out isomorphic duplicates. The `run_sampler` function generates k random models at each domain size and computes satisfiability rates with Wilson score confidence intervals. Helper functions format results into tables and generate plots of formula length vs. smallest model size.

**Outcome**: Produces detailed statistical analysis showing how satisfiability rates change with domain size, empirically revealing whether formulas exhibit the 0-1 law (rates converging to 0 or 1 as n increases).

### `benchmarks.py`
**Goal**: Define test formulas for experimentation.

**Implementation**: Declares predicate symbols (C, E, D, P) and contains a dictionary of benchmark formulas expressed using the formula AST. Most formulas are commented out; the active one tests whether uncolored nodes only connect to colored nodes.

**Outcome**: Provides concrete FO² formulas to test the system's satisfiability analysis capabilities.

### `config.py`
**Goal**: Centralize configuration parameters for experiments.

**Implementation**: Defines constants for reproducibility (random seed), exhaustive search limits (max domain size n=5), isomorphism filtering threshold (n≤5), sampler parameters (domain sizes [8,10,15,20,50,100] with k=1000 samples), and probability distributions for random model generation (p_edge=0.5, p_unary=0.5).

**Outcome**: Allows easy tuning of experimental parameters without modifying core logic.

### `main.py`
**Goal**: Orchestrate the complete experimental pipeline.

**Implementation**: For each benchmark formula, prints a pre-flight manifest showing formula properties and execution plan, runs both exhaustive search and Monte Carlo sampling experiments, and displays results in formatted tables. Calculates Wilson confidence intervals and generates a plot of smallest model sizes versus formula lengths.

**Outcome**: Produces a comprehensive report for each formula showing exact satisfiability counts (small n), estimated rates with confidence intervals (large n), and visualizations—demonstrating whether the 0-1 law holds empirically.
