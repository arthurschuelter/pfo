# --------------------------------------------------------
from typing import Tuple

import numpy as np
import pyswarms as ps

from pfo_base import PFOBase

# --------------------------------------------------------


class HP3DParticleSwarm:
    """Particle Swarm Optimization for HP3D protein folding"""

    def __init__(self, pfo_base: PFOBase):
        self.pfo_base = pfo_base
        self.energy_history: list[float] = []
        self.best_energy_value = float("inf")
        self.best_conformation = None
        self.best_moves: np.ndarray
        self.get_results_summary = pfo_base.get_results_summary

    def set_parameters(self):
        self.n_particles = 300
        self.n_iterations = 10000

        self.options = {
            "c1":   2.05,     # Cognitive parameter
            "c2":   2.05,     # Social parameter
            "w":    0.70,     # Inertia weight
            "k":    3.00,     # Number of neighbors
            "p":    2.00,     # Minkowski distance
        }


    def fitness_wrapper(self, particles: np.ndarray) -> np.ndarray:
        """
        Wrapping fitness function to work with PySwarms

        Args:
            particles: Array of shape (n_particles, dimensions)
                      Each row is a particle representing a move sequence

        Returns:
            Array of fitness values for each particle
        """
        n_particles = particles.shape[0]
        fitness_values = np.zeros(n_particles)

        for i in range(n_particles):
            discrete_moves = self.continuous_to_discrete_deterministic(particles[i])
            # discrete_moves = self.continuous_to_discrete_probabilistic(particles[i])

            fitness_values[i] = self.evaluate_moves(discrete_moves)

        return fitness_values
    
    def continuous_to_discrete_probabilistic(self, continuous_values: np.ndarray) -> np.ndarray:
        """
        Map continuous [0, 5] to discrete {0,1,2,3,4,5} probabilistically.
        This preserves gradient information that pure rounding destroys.
        
        Example: 
            2.7 → 70% chance of 3, 30% chance of 2
            2.3 → 30% chance of 3, 70% chance of 2
        
        This allows PSO's continuous optimization to work on discrete problems!
        """
        # Ensure values are in valid range
        continuous_values = np.clip(continuous_values, 0, 5)
        
        # Get floor and fractional parts
        floor_values = np.floor(continuous_values).astype(int)
        fractions = continuous_values - floor_values
        
        # Probabilistic rounding based on fractional part
        random_vals = np.random.random(len(continuous_values))
        discrete_moves = np.where(random_vals < fractions, floor_values + 1, floor_values)
        
        # Final clipping to ensure [0, 5]
        discrete_moves = np.clip(discrete_moves, 0, 5)
        
        return discrete_moves

    def continuous_to_discrete_deterministic(self, continuous_values: np.ndarray) -> np.ndarray:
        """
        Deterministic mapping for final best solution extraction.
        Uses standard rounding.
        """
        discrete_moves = np.round(continuous_values).astype(int)
        discrete_moves = np.clip(discrete_moves, 0, 5)
        return discrete_moves


    def evaluate_moves(self, moves: np.ndarray) -> float:
        """
        Evaluate a single move sequence

        Args:
            moves: Array of move directions (0-5)

        Returns:
            Energy value (fitness to minimize)
        """
        self.pfo_base.evaluation_count += 1

        # Generate conformation
        conformation = self.pfo_base.moves_to_conformation(moves)

        if conformation is None:
            # Invalid conformation - return high penalty
            return 0

        # Calculate energy
        energy = self.pfo_base.calculate_energy(conformation)

        # Update best solution
        if energy < self.best_energy_value:
            self.best_energy_value = energy
            self.best_conformation = conformation.copy()
            self.best_moves = moves.copy()

            # Update base model's best solution
            if energy < self.pfo_base.best_energy:
                self.pfo_base.best_energy = energy
                self.pfo_base.best_conformation = conformation.copy()

        return energy

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization

        Returns:
            Tuple of (best_moves, best_energy)
        """
        # Define bounds: each dimension (move) can be 0-5
        dimensions = self.pfo_base.length - 1
        bounds = (np.zeros(dimensions), np.full(dimensions, 5))

        # Initialize optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=dimensions,
            options=self.options,
            bounds=bounds,
        )

        # Run optimization with callback to track history
        cost, pos = optimizer.optimize(
            self.fitness_wrapper, iters=self.n_iterations, verbose=True
        )

        # Store energy history from optimizer
        self.energy_history = optimizer.cost_history

        # Convert best position to discrete moves
        best_moves = np.round(pos).astype(int)
        best_moves = np.clip(best_moves, 0, 5)

        print("\nPSO Optimization Complete!")
        print(f"Best Energy: {self.best_energy_value}")
        print(f"H-H Contacts: {-int(self.best_energy_value)}")
        print(f"Total Evaluations: {self.pfo_base.evaluation_count}")

        return best_moves, self.best_energy_value

    def get_pso_convergence_data(self):
        """Get convergence data for PSO"""
        iterations = list(range(len(self.energy_history)))
        return iterations, self.energy_history

    def plot_convergence(self):
        """Plot PSO convergence history"""
        if len(self.energy_history) > 0:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(self.energy_history, linewidth=2)
            plt.xlabel("Iteration")
            plt.ylabel("Best Energy")
            plt.title("PSO Convergence History")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
