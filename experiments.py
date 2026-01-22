import math
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from tabulate import tabulate
import networkx as nx

import config
from formula import Formula, get_predicates
from generate_models import enumerate_models, sample_model
from model import Model


def _bits_to_edge_set(bits: int, n: int) -> set[tuple[int, int]]:
    edges = set()
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (bits >> idx) & 1:
                edges.add((i, j))
            idx += 1
    return edges

def _model_to_str(model: Model | None) -> str:
    """Formats a model's interpretation into a concise string for display."""
    if model is None:
        return "N/A"
    
    parts = []
    # Sort by predicate symbol for consistent output
    for pred_symbol, relation in sorted(model.interpretation.items()):
        if not relation:
            continue  # Don't print empty relations
        
        # Format tuples nicely for readability
        relation_str = ", ".join(
            str(t[0]) if len(t) == 1 else str(t) for t in sorted(list(relation))
        )
        parts.append(f"I({pred_symbol})={{{relation_str}}}")
        
    if not parts:
        return "{empty interpretation}"
        
    return ", ".join(parts)


def _get_canonical_hash(model: 'Model', domain_size: int, predicates: set['Predicate']) -> str:
    """
    Computes a canonical hash for a model using graph isomorphism checks.
    """
    G = nx.Graph()
    G.add_nodes_from(range(domain_size))
    
    unary_preds = sorted([p for p in predicates if p.arity == 1], key=lambda p: p.symbol)
    binary_preds = [p for p in predicates if p.arity == 2]

    # Encode the truth values of **all** unary predicates into a single
    # hashable string stored under one attribute per node.
    for node in G.nodes:
        bits = [
            '1' if (node,) in model.interpretation.get(pred.symbol, set()) else '0'
            for pred in unary_preds
        ]
        # The resulting bit‑string is hashable, so WL hashing accepts it.
        G.nodes[node]['unary_sig'] = ''.join(bits) if bits else '0'

    # Add edges for the binary predicate
    if binary_preds:
        edge_relation = model.interpretation.get(binary_preds[0].symbol, set())
        G.add_edges_from(edge for edge in edge_relation if edge[0] < edge[1])

    # Use the Weisfeiler-Lehman hash for canonical representation
    return nx.weisfeiler_lehman_graph_hash(
        G,
        node_attr='unary_sig' if unary_preds else None
    )

def run_exhaustive(formula: Formula) -> Dict[str, Any]:
    """
    Performs an exhaustive search for a satisfying model up to N_MAX_EXHAUSTIVE.
    Includes isomorphism filtering for n <= N_MAX_ISOMORPHISM_FILTER.
    """
    predicates = get_predicates(formula)
    results_by_n = []
    first_model_size = None

    for n in range(2, config.N_MAX_EXHAUSTIVE + 1):
        models_iter = enumerate_models(n, predicates)
        
        total_labeled = 0
        sat_labeled = 0
        first_sat_labeled_model = None
        first_unsat_labeled_model = None
        
        # Isomorphism filtering data
        is_iso_filtered = n <= config.N_MAX_ISOMORPHISM_FILTER
        seen_hashes = set()
        total_iso = 0
        sat_iso = 0
        first_sat_iso_model = None
        first_unsat_iso_model = None

        for model in models_iter:
            total_labeled += 1
            is_satisfying = model.eval(formula)

            if is_satisfying:
                sat_labeled += 1
                if first_sat_labeled_model is None:
                    first_sat_labeled_model = model
            else:  # Not satisfying
                if first_unsat_labeled_model is None:
                    first_unsat_labeled_model = model

            if is_iso_filtered:
                graph_hash = _get_canonical_hash(model, n, predicates)
                if graph_hash not in seen_hashes:
                    seen_hashes.add(graph_hash)
                    total_iso += 1
                    if is_satisfying:
                        sat_iso += 1
                        if first_sat_iso_model is None:
                            first_sat_iso_model = model
                    else: # Not satisfying
                        if first_unsat_iso_model is None:
                            first_unsat_iso_model = model
            
            if is_satisfying and first_model_size is None:
                first_model_size = n
        
        results_by_n.append({
            "n": n,
            "total_labeled": total_labeled,
            "satisfying_labeled": sat_labeled,
            "is_iso_filtered": is_iso_filtered,
            "total_isomorphic": total_iso,
            "satisfying_isomorphic": sat_iso,
            "first_sat_labeled_model": first_sat_labeled_model,
            "first_unsat_labeled_model": first_unsat_labeled_model,
            "first_sat_iso_model": first_sat_iso_model,
            "first_unsat_iso_model": first_unsat_iso_model,
        })
            
    return {"first_model_size": first_model_size, "details": results_by_n}

def run_sampler(formula: Formula) -> Dict[int, Dict[str, Any]]:
    """Approximates satisfiability rates using random sampling."""
    predicates = get_predicates(formula)
    results = {}

    for n in config.SAMPLER_N_LIST:
        sat_count = sum(1 for _ in range(config.SAMPLER_K) if sample_model(n, predicates).eval(formula))
        
        p_hat = sat_count / config.SAMPLER_K
        z = 1.96 # 95% confidence
        n_eff = config.SAMPLER_K
        
        try:
            wilson_num = p_hat + z*z/(2*n_eff)
            wilson_den = 1 + z*z/n_eff
            wilson_err = z * math.sqrt(p_hat*(1-p_hat)/n_eff + z*z/(4*n_eff*n_eff))
            center = wilson_num / wilson_den
            radius = wilson_err / wilson_den
            conf_interval = (f"{max(0, center - radius):.3f}", f"{min(1, center + radius):.3f}")
        except ZeroDivisionError:
            conf_interval = (0.0, 1.0)

        results[n] = {'rate': f"{p_hat:.3f}", 'conf_interval': conf_interval, 'k': config.SAMPLER_K}

    return results

def plot_size_vs_length(results: List[Dict[str, Any]], title: str = "Model Size vs. Formula Length"):
    """Plots smallest model size against formula length."""
    lengths = [r['len'] for r in results]
    sizes = [r['size'] for r in results]

    plt.figure()
    plt.semilogy(lengths, sizes, 'o')
    plt.xlabel("Formula Length (string length)")
    plt.ylabel("Smallest Model Size (log scale)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def print_results_table(formula_name: str, formula: Formula, exhaustive_results: Dict):
    """Prints a summary table of the results."""
    print("\n--- Exhaustive Search Results ---")
    ex_details = exhaustive_results['details']
    
    table_data = []
    for row in ex_details:
        n = row['n']
        total_l = row['total_labeled']
        sat_l = row['satisfying_labeled']
        unsat_l = total_l - sat_l

        if row['is_iso_filtered']:
            total_i = row['total_isomorphic']
            sat_i = row['satisfying_isomorphic']
            unsat_i = total_i - sat_i
            labeled_str = f"{total_l:,} ({total_i:,})"
            sat_str = f"{sat_l:,} ({sat_i:,})"
            unsat_str = f"{unsat_l:,} ({unsat_i:,})"
        else:
            labeled_str = f"{total_l:,}"
            sat_str = f"{sat_l:,}"
            unsat_str = f"{unsat_l:,}"
        
        # Add model examples for small n, as requested
        if n <= 4:
            if row['is_iso_filtered']:
                sat_model_example = _model_to_str(row['first_sat_iso_model'])
                unsat_model_example = _model_to_str(row['first_unsat_iso_model'])
            else:
                sat_model_example = _model_to_str(row['first_sat_labeled_model'])
                unsat_model_example = _model_to_str(row['first_unsat_labeled_model'])
            
            sat_str += f"\nEx: {sat_model_example}"
            unsat_str += f"\nEx: {unsat_model_example}"

        table_data.append([n, labeled_str, sat_str, unsat_str])

    headers = ["n", "Labeled Models Tested (non-isomorphic)", "Satisfying", "Unsatisfying"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    if exhaustive_results['first_model_size']:
        print(f"\nSmallest satisfying model found at n = {exhaustive_results['first_model_size']}.")
    else:
        print(f"\nNo satisfying model found up to n = {config.N_MAX_EXHAUSTIVE}.")

def print_sampler_results(sampler_results: Dict):
    """Prints a table of the sampler results."""
    print("\n--- Random Sampler Results ---")
    samp_headers = ["Domain Size (n)", "Sampled Models (k)", "Satisfiability Rate", "95% Wilson Interval"]
    samp_table = [
        [n, res['k'], res['rate'], f"[{res['conf_interval'][0]}, {res['conf_interval'][1]}]"]
        for n, res in sampler_results.items()
    ]
    print(tabulate(samp_table, headers=samp_headers, tablefmt="grid"))
