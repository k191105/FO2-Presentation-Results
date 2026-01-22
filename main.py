import random
import math
from tabulate import tabulate

from experiments import run_exhaustive, run_sampler, plot_size_vs_length, print_results_table, print_sampler_results
from benchmarks import BENCHMARKS
from formula import get_predicates
from generate_models import count_models
import config

def _calculate_wilson_half_width(p, n, z=1.96):
    """Calculates the half-width of the Wilson score interval."""
    try:
        n_eff = n
        wilson_num = p + z*z/(2*n_eff)
        wilson_den = 1 + z*z/n_eff
        wilson_err = z * math.sqrt(p*(1-p)/n_eff + z*z/(4*n_eff*n_eff))
        radius = wilson_err / wilson_den
        return radius
    except ZeroDivisionError:
        return 0.5

def main():
    """
    Runs the main experiment to test satisfiability of benchmark formulas.
    """
    random.seed(config.SEED)

    print("Starting FO₂ Playground experiments.")
    print(f"Reproducibility seed set to {config.SEED}.")
    print(f"Exhaustive search will run up to n={config.N_MAX_EXHAUSTIVE}.")
    print(f"Isomorphism filtering will run up to n={config.N_MAX_ISOMORPHISM_FILTER}.")
    
    plot_data = []
    
    for name, formula in BENCHMARKS.items():
        # --- Pre-flight Manifest ---
        predicates = get_predicates(formula)
        num_unary = sum(1 for p in predicates if p.arity == 1)
        num_binary = sum(1 for p in predicates if p.arity == 2)

        print("\n" + "="*80)
        print(f"EXPERIMENT: {name}")
        print(f"φ = {formula}")
        print(f"  - String length: {len(str(formula))}")
        print(f"  - Predicates: {num_unary} unary, {num_binary} binary")
        
        print("\nExecution Plan (Exhaustive Search):")
        plan_headers = ["Domain Size (n)", "Number of Labeled Models"]
        plan_table = [[n, f"{count_models(n, predicates):,}"] for n in range(2, config.N_MAX_EXHAUSTIVE + 1)]
        print(tabulate(plan_table, headers=plan_headers))
        
        print("\nExecution Plan (Random Sampler):")
        sampler_headers = ["Domain Size (n)", "Samples (k)", "Expected Wilson Half-Width (at p=0.5)"]
        sampler_plan = [[
            n, config.SAMPLER_K, f"±{_calculate_wilson_half_width(0.5, config.SAMPLER_K):.3f}"
        ] for n in config.SAMPLER_N_LIST]
        print(tabulate(sampler_plan, headers=sampler_headers))
        print("-" * 80)

        # --- Running Experiments ---
        exhaustive_results = run_exhaustive(formula)
        sampler_results = run_sampler(formula)
        
        # --- Reporting Results ---
        print_results_table(name, formula, exhaustive_results)
        print_sampler_results(sampler_results)
        print("="*80)
        
        if exhaustive_results['first_model_size']:
            plot_data.append({"len": len(str(formula)), "size": exhaustive_results['first_model_size']})

    # --- Final Plot ---
    if plot_data:
        print("\nPlotting formula lengths vs. smallest model sizes...")
        plot_size_vs_length(plot_data)
        print("Plot generated. Close the plot window to exit.")
    else:
        print("\nNo models found for any benchmark formulas, so no plot will be generated.")

if __name__ == "__main__":
    main()
