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


def optimize_ga(sequence: str, seq_name: str = "") -> PFOBase:
    print("Optimizing with Genetic Algorithm...")
    start = time.time()

    hp_model = PFOBase(sequence, "Genetic Algorithm", seq_name, "ga")
    # hp_model.print_header()
    ga_model = instantiate_ga(hp_model)

    ga_model.run(function=hp_model.fitness_function, no_plot=True)
    hp_model.energy_history = ga_model.report

    end = time.time()
    if hp_model.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (GA)")

    # hp_model.get_results_summary("GENETIC ALGORITHM")

    hp_model.time = end - start
    return hp_model


def instantiate_ga(hp_model: PFOBase):
    # default_params = {
    #     "max_num_iteration": 10000,
    #     "population_size": 300,
    #     "mutation_probability": 0.2,
    #     "elit_ratio": 0.08,
    #     "parents_portion": 0.45,
    #     "crossover_type": "two_point",
    #     "mutation_type": "uniform_by_center",
    #     "selection_type": "tournament",
    #     "max_iteration_without_improv": None,
    # }

    default_params = {
        "max_num_iteration": 20000,
        "population_size": 500,
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


def optimize_sa(sequence, seq_name: str = ""):
    print("Optimizing with Simulated Annealing...")
    start = time.time()

    hp_model = PFOBase(sequence, "Simulated Annealing", seq_name, "sa")
    # hp_model.print_header()

    sa_solver = HP3DSimulatedAnnealing(hp_model)
    sa_solver.set_parameters()
    best_state, best_energy = sa_solver.optimize()

    # sa_solver.get_results_summary("SIMULATED ANNEALING")

    hp_model.best_energy = sa_solver.best_energy_value
    hp_model.energy_history = sa_solver.energy_history

    end = time.time()
    if sa_solver.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (SA)")
        pass

    hp_model.time = end - start
    return hp_model


def optimize_pso(sequence, seq_name: str = ""):
    print("Optimizing with Particle Swarm Optimization...")
    start = time.time()
    hp_model = PFOBase(sequence, "Particle Swarm Optimization", seq_name, "pso")
    # hp_model.print_header()

    pso_solver = HP3DParticleSwarm(hp_model)
    pso_solver.set_parameters()
    best_moves, best_energy = pso_solver.optimize()

    # pso_solver.get_results_summary("PARTICLE SWARM OPTIMIZATION")

    hp_model.best_energy = pso_solver.best_energy_value
    hp_model.energy_history = pso_solver.energy_history

    end = time.time()
    if pso_solver.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (PSO)")

    hp_model.time = end - start
    return hp_model


def compare_algorithms(sequence, label):
    """Compare GA, SA, and PSO performance"""
    print(f"\n{'='*80}")
    print(f"COMPARING ALGORITHMS FOR SEQUENCE: {label} ({sequence})")
    print(f"{'='*80}")

    # GA
    print("\n" + "-" * 40)
    ga_result = optimize_ga(sequence, label)
    ga_energy = ga_result.best_energy
    ga_evaluations = ga_result.evaluation_count
    ga_time = f"{ga_result.time:.2f}" + " s"

    # SA
    print("\n" + "-" * 40)
    sa_result = optimize_sa(sequence, label)
    sa_energy = sa_result.best_energy
    sa_evaluations = sa_result.evaluation_count
    sa_time = f"{sa_result.time:.2f}" + " s"

    # PSO
    print("\n" + "-" * 40)
    pso_result = optimize_pso(sequence, label)
    pso_energy = pso_result.best_energy
    pso_evaluations = pso_result.evaluation_count
    pso_time = f"{pso_result.time:.2f}" + " s"
    

    # Comparison summary
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Sequence: {label}")
    print(f"Length: {len(sequence)}")
    print()
    print(
        f"{'Algorithm':<15} {'Best Energy':<12} {'H-H Contacts':<12} {'Evaluations':<12} {'Execution Time':<16}"
    )
    print("-" * 80)
    print(f"{'GA':<15} {ga_energy:<12.2f} {-int(ga_energy):<12} {ga_evaluations:<12} {ga_time:<16}")
    print(f"{'SA':<15} {sa_energy:<12.2f} {-int(sa_energy):<12} {sa_evaluations:<12} {sa_time:<16}")
    print(
        f"{'PSO':<15} {pso_energy:<12.2f} {-int(pso_energy):<12} {pso_evaluations:<12} {pso_time:<16}"
    )
    original_stdout = sys.stdout
    with open('logs.txt', 'a') as f:
        sys.stdout = f
        print(f"\n{'='*80}")
        print("ALGORITHM COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Sequence: {label}")
        print(f"Length: {len(sequence)}")
        print()
        print(
            f"{'Algorithm':<15} {'Best Energy':<12} {'H-H Contacts':<12} {'Evaluations':<12} {'Execution Time':<16}"
        )
        print("-" * 80)
        print(f"{'GA':<15} {ga_energy:<12.2f} {-int(ga_energy):<12} {ga_evaluations:<12} {ga_time:<16}")
        print(f"{'SA':<15} {sa_energy:<12.2f} {-int(sa_energy):<12} {sa_evaluations:<12} {sa_time:<16}")
        print(
            f"{'PSO':<15} {pso_energy:<12.2f} {-int(pso_energy):<12} {pso_evaluations:<12} {pso_time:<16}"
        )
        print()

    sys.stdout = original_stdout
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
        # "PHHHH", # 1PLW
        # "PHPPHHPPPPHHPHPPPHPPPPPHPPPPPPPHPPPHHHPPHPHPHHHPPPPHPH", # 1ENH
        
        "HHPPPHHHPPHPPP", # 2MI1 (Somatostatin)
        "PPPHPHPPPPPPPHPPPPHPPHHPHHHPP", # 1GCN (Glucagon)
        "PPPPHPHHHPPPHPHPPHHHPHPHHPHPPPHPHHHHHHPPHHPPHP", # 1CRN
        
        # "HPHPPHHPHPPHPHHPPHPH", # 3d1
        # "HHPPHPPHPPHPPHPPHPPHPPHH", # 3d2
        # "PPHPPHHPPPPHHPPPPHHPPPPHH", # 3d3
        # "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", # 3d4
        # "PPHHHPHHHPPPHPHHPHHPPHPHHHHPHPPHHHHHPHPHHPPHHP", # 3d5

        # "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH", # 3d6
        # "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH", # 3d7
    ]

    names = [
        # "1PLW",
        # "1ENH",
        "2MI1 (Somatostatin)",
        "1GCN (Glucagon)",
        "1CRN (Crambin)",

        # "3d1", # Min: -11 | Len: 20 
        # "3d2", # Min: -13 | Len: 24
        # "3d3", # Min: -9  | Len: 25
        # "3d4", # Min: -18 | Len: 36
        # "3d5", # Min: -35 | Len: 46

        # "3d6", # Min: -31 | Len: 48
        # "3d7", # Min: -34 | Len: 50
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
        h[0] = np.float64(0)
        h = np.array(h, dtype=float)
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

    save_path = f"./img/0.benchmark/{name}-convergence.png"
    plt.savefig(save_path)
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
