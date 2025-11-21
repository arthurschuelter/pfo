# python3 -m venv env
# source env/bin/activate
# pip3 install -r requirements.txt
# python3 src/main.py
# pip3 freeze > requirements.txt
# deactivate
# --------------------------------------------------------
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

from pfo_base import PFOBase
from pfo_pso import HP3DParticleSwarm
from pfo_sa import HP3DSimulatedAnnealing

# --------------------------------------------------------


def optimize_ga(sequence: str) -> PFOBase:
    print("Optimizing with Genetic Algorithm...")
    hp_model = PFOBase(sequence, "Genetic Algorithm")
    hp_model.print_header()
    ga_model = instantiate_ga(hp_model)

    ga_model.run(function=hp_model.fitness_function, no_plot=True)
    hp_model.energy_history = ga_model.report

    if hp_model.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (GA)")

    hp_model.get_results_summary("GENETIC ALGORITHM")

    return hp_model


def instantiate_ga(hp_model: PFOBase):
    default_params = {
        "max_num_iteration": 10000,
        "population_size": 300,
        "mutation_probability": 0.2,
        "elit_ratio": 0.08,
        "parents_portion": 0.45,
        "crossover_type": "two_point",
        "mutation_type": "uniform_by_center",
        "selection_type": "tournament",
        "max_iteration_without_improv": None,
    }

    return ga(
        dimension=hp_model.length - 1,
        variable_type="int",
        variable_boundaries=hp_model.var_bound,
        algorithm_parameters=default_params,
    )


def optimize_sa(sequence):
    print("Optimizing with Simulated Annealing...")
    start = time.time()

    hp_model = PFOBase(sequence, "Simulated Annealing")
    hp_model.print_header()

    sa_solver = HP3DSimulatedAnnealing(hp_model)
    sa_solver.set_parameters()
    best_state, best_energy = sa_solver.optimize()

    sa_solver.get_results_summary("SIMULATED ANNEALING")

    hp_model.best_energy = sa_solver.best_energy_value
    hp_model.energy_history = sa_solver.energy_history

    print(f"Used time: {(time.time() - start):.2f} seconds")
    if sa_solver.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (SA)")
        pass

    return hp_model


def optimize_pso(sequence):
    print("Optimizing with Particle Swarm Optimization...")
    start = time.time()
    hp_model = PFOBase(sequence, "Particle Swarm Optimization")
    hp_model.print_header()

    pso_solver = HP3DParticleSwarm(hp_model)
    pso_solver.set_parameters()
    best_moves, best_energy = pso_solver.optimize()

    pso_solver.get_results_summary("PARTICLE SWARM OPTIMIZATION")

    hp_model.best_energy = pso_solver.best_energy_value
    hp_model.energy_history = pso_solver.energy_history

    print(f"Used time: {(time.time() - start):.2f} seconds")
    if pso_solver.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (PSO)")

    return hp_model


def compare_algorithms(sequence, label):
    """Compare GA, SA, and PSO performance"""
    print(f"\n{'='*80}")
    print(f"COMPARING ALGORITHMS FOR SEQUENCE: {sequence}")
    print(f"{'='*80}")

    # GA
    print("\n" + "-" * 40)
    ga_result = optimize_ga(sequence)
    ga_energy = ga_result.best_energy
    ga_evaluations = ga_result.evaluation_count

    # SA
    print("\n" + "-" * 40)
    sa_result = optimize_sa(sequence)
    sa_energy = sa_result.best_energy
    sa_evaluations = sa_result.evaluation_count

    # PSO
    print("\n" + "-" * 40)
    pso_result = optimize_pso(sequence)
    pso_energy = pso_result.best_energy
    pso_evaluations = pso_result.evaluation_count

    # Comparison summary
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Sequence: {label}")
    print(f"Length: {len(sequence)}")
    print()
    print(
        f"{'Algorithm':<15} {'Best Energy':<12} {'H-H Contacts':<12} {'Evaluations':<12}"
    )
    print("-" * 80)
    print(f"{'GA':<15} {ga_energy:<12.2f} {-int(ga_energy):<12} {ga_evaluations:<12}")
    print(f"{'SA':<15} {sa_energy:<12.2f} {-int(sa_energy):<12} {sa_evaluations:<12}")
    print(
        f"{'PSO':<15} {pso_energy:<12.2f} {-int(pso_energy):<12} {pso_evaluations:<12}"
    )
    print()

    # Find best algorithm
    results = [("GA", ga_energy), ("SA", sa_energy), ("PSO", pso_energy)]
    best_algo = min(results, key=lambda x: x[1])

    print(f"🏆 {best_algo[0]} found the best solution with energy {best_algo[1]:.2f}!")
    print(f"{'='*80}")

    return [ga_result, sa_result, pso_result]


def main():
    method = sys.argv[1]

    sequences = [
        # "PHHHH",
        # "PPPPHPHHHPPPHPHPPHHHPHPHHPHPPPHPHHHHHHPPHHPPHP",
        "PHPPHHPPPPHHPHPPPHPPPPPHPPPPPPPHPPPHHHPPHPHPHHHPPPPHPH",
    ]

    names = [
        # "1PLW",
        # "1CRN",
        "1ENH",
    ]

    for i, sequence in enumerate(sequences):
        models = []
        labels = []

        if method == "ga":
            ga_model = optimize_ga(sequence)
            models.append(ga_model.energy_history)
            labels.append(ga_model.label)
        elif method == "sa":
            sa_model = optimize_sa(sequence)
            models.append(sa_model.energy_history)
            labels.append(sa_model.label)
        elif method == "pso":
            pso_model = optimize_pso(sequence)
            models.append(pso_model.energy_history)
            labels.append(pso_model.label)
        elif method == "compare":
            compare_models = compare_algorithms(sequence, names[i])
            models = [m.energy_history for m in compare_models]
            labels = [m.label for m in compare_models]

        plot_energy_history(models, labels, names[i])


def plot_energy_history(history, labels, name):
    for h, l in zip(history, labels):
        h[0] = 0
        plt.plot(h, label=l)

    plt.xlabel("Generation")
    plt.ylabel("Energy")
    plt.title(f"Convergence for {name}")
    plt.legend(loc="best")

    y_min = min(min(h) for h in history)
    y_max = 0
    plt.ylim(y_min - 1, y_max)
    plt.axhline(y=y_min, linestyle="--", linewidth=1, alpha=0.7)

    ticks = plt.gca().get_yticks()

    if y_min not in ticks:
        ticks = np.append(ticks, y_min)

    ticks = np.sort(ticks)
    plt.gca().set_yticks(ticks)

    plt.show()


if __name__ == "__main__":
    main()
