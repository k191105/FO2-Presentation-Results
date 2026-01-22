import itertools
import random
from typing import Iterable

from tqdm import tqdm

from formula import Predicate
from model import Model
import config


def count_models(domain_size: int, predicates: set[Predicate]) -> int:
    """
    Calculates the number of unique labeled models for a given domain size.
    This assumes at most one binary predicate, treated as an undirected graph.
    """
    num_binary = sum(1 for p in predicates if p.arity == 2)
    if num_binary > 1:
        raise ValueError("Model counting currently supports at most one binary predicate.")

    n = domain_size
    num_unary = sum(1 for p in predicates if p.arity == 1)

    # Number of undirected graphs on n labeled vertices is 2^(nC2).
    num_graphs = 2**(n * (n - 1) // 2) if num_binary > 0 else 1
    # For each unary predicate, there are 2^n ways to color n vertices.
    num_colorings = (2**n)**num_unary
    
    return num_graphs * num_colorings


def enumerate_models(domain_size: int, predicates: set[Predicate]) -> Iterable[Model]:
    """
    Exhaustively yields every labeled model for a given domain size and signature.
    This generator assumes at most one binary predicate, which is treated as an
    undirected graph. It will raise an error if more than one is provided.
    """
    if domain_size > config.N_MAX_EXHAUSTIVE:
        raise ValueError(f"Enumeration is only supported for domain_size <= {config.N_MAX_EXHAUSTIVE}")

    unary_predicates = {p for p in predicates if p.arity == 1}
    binary_predicates = {p for p in predicates if p.arity == 2}
    if len(binary_predicates) > 1:
        raise ValueError("Model enumeration currently supports at most one binary predicate.")

    domain = set(range(domain_size))
    vertices = list(range(domain_size))

    # Generate graph structures only if a binary predicate exists.
    if binary_predicates:
        possible_edges = list(itertools.combinations(vertices, 2))
        graph_edge_sets = [
            set(itertools.chain.from_iterable(((u, v), (v, u)) for u, v in edge_set))
            for i in range(len(possible_edges) + 1)
            for edge_set in itertools.combinations(possible_edges, i)
        ]
    else:
        graph_edge_sets = [set()]  # Only one structure: no edges

    # Generate all colorings for each unary predicate
    vertex_powerset = [
        set(subset)
        for i in range(len(vertices) + 1)
        for subset in itertools.combinations(vertices, i)
    ]
    unary_colorings = list(itertools.product(vertex_powerset, repeat=len(unary_predicates)))
    
    total_models = len(graph_edge_sets) * len(unary_colorings)
    pbar = tqdm(total=total_models, desc=f"Generating models for n={domain_size}", leave=False)

    for graph_edges in graph_edge_sets:
        for coloring_combo in unary_colorings:
            interpretation = {}
            if binary_predicates:
                first_binary_pred = next(iter(binary_predicates))
                interpretation[first_binary_pred.symbol] = graph_edges

            for i, pred in enumerate(unary_predicates):
                interpretation[pred.symbol] = {(elem,) for elem in coloring_combo[i]}
            
            pbar.update(1)
            yield Model(domain, interpretation)
    pbar.close()


def sample_model(domain_size: int, predicates: set[Predicate]) -> Model:
    """
    Returns a single random model instance.
    """
    domain = set(range(domain_size))
    interpretation = {}

    # This sampler can handle multiple binary predicates, unlike the enumerator.
    binary_predicates = {p for p in predicates if p.arity == 2}
    unary_predicates = {p for p in predicates if p.arity == 1}

    for pred in unary_predicates:
        relation = {(v,) for v in domain if random.random() < config.P_UNARY}
        interpretation[pred.symbol] = relation

    for pred in binary_predicates:
        relation = set()
        # Assume undirected graphs for all binary relations in the sampler
        for u, v in itertools.combinations(domain, 2):
            if random.random() < config.P_EDGE:
                relation.add((u, v))
                relation.add((v, u))
        interpretation[pred.symbol] = relation
    
    return Model(domain, interpretation)
