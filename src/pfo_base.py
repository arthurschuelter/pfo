# --------------------------------------------------------
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------


class PFOBase:
    def __init__(self, sequence: str, label: str = ""):
        """
        Initialize 3D HP lattice model for genetic algorithm

        Args:
            sequence: Protein sequence using 'H' (hydrophobic) and 'P' (polar)
        """
        self.sequence = sequence.upper()
        self.length = len(sequence)
        self.evaluation_count = 0
        self.label = label

        self.best_conformation: Any = None
        self.best_energy = float("inf")
        self.energy_history: list[float] = []
        self.var_bound = [(0, 5)] * (self.length - 1)  # n-1 moves for n residues

        # [TO DO] aminoacids to HP
        if not all(aa in "HP" for aa in self.sequence):
            raise ValueError("Sequence must contain only 'H' and 'P' residues")

        # 6 directions in 3D cubic lattice
        self.directions = np.array(
            [
                [1, 0, 0],  # 0: +x
                [-1, 0, 0],  # 1: -x
                [0, 1, 0],  # 2: +y
                [0, -1, 0],  # 3: -y
                [0, 0, 1],  # 4: +z
                [0, 0, -1],  # 5: -z
            ]
        )

    def fitness_function(self, moves: np.ndarray) -> float:
        """
        Fitness function to be minimized

        Args:
            moves: Array of move directions (0-5) for each step

        Returns:
            Fitness value (energy to minimize)
        """
        self.evaluation_count += 1

        # Convert continuous GA values to discrete moves
        discrete_moves = np.round(moves).astype(int)
        discrete_moves = np.clip(discrete_moves, 0, 5)

        # Generate conformation from moves
        conformation = self.moves_to_conformation(discrete_moves)

        if conformation is None:
            return 0

        energy = self.calculate_energy(conformation)

        total_fitness = energy

        if total_fitness < self.best_energy:
            self.best_energy = total_fitness
            self.best_conformation = conformation.copy()

        return total_fitness

    def moves_to_conformation(self, moves: np.ndarray) -> Optional[np.ndarray]:
        """Convert move sequence to 3D conformation"""
        conformation = np.zeros((self.length, 3))
        conformation[0] = [0, 0, 0]

        occupied = {tuple(conformation[0])}

        for i, move in enumerate(moves):
            new_pos = conformation[i] + self.directions[move]

            if tuple(new_pos) in occupied:
                # return np.array([0] * self.length)
                return None

            conformation[i + 1] = new_pos
            occupied.add(tuple(new_pos))

        return conformation

    def calculate_energy(self, conformation: np.ndarray) -> float:
        hh_contacts = 0

        for i in range(self.length):
            for j in range(i + 2, self.length):
                if self.sequence[i] == "H" and self.sequence[j] == "H":
                    if self.are_neighbors(conformation[i], conformation[j]):
                        hh_contacts += 1

        return -hh_contacts

    def are_neighbors(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        distance = np.linalg.norm(pos1 - pos2)
        return bool(abs(distance - 1.0) < 0.1)

    def get_results_summary(self, title):
        """Print summary of optimization results"""
        print("\n" + "=" * 60)
        print(f"{title} OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Sequence: {self.sequence}")
        print(f"Length: {self.length}")
        print(f"Search space: 6^{self.length-1} = {6**(self.length-1):.2e}")
        print(f"Total function evaluations: {self.evaluation_count}")

        print("=" * 60)

    def visualize_best(self, title: str = "Best 3D HP Conformation") -> None:
        """Visualize the best conformation found"""
        best_conformation = self.best_conformation
        if best_conformation is None:
            print("No valid conformation found yet. Run optimization first.")
            return

        energy = self.calculate_energy(best_conformation)
        hh_contacts = -int(energy)

        fig: Figure = plt.figure(figsize=(12, 10))
        ax: Axes3D = fig.add_subplot(111, projection="3d")  # Type hint as Axes3D

        # Plot backbone
        ax.plot(
            best_conformation[:, 0],
            best_conformation[:, 1],
            best_conformation[:, 2],
            "k-",
            linewidth=3,
            alpha=0.7,
            label="Backbone",
        )

        for i, (x, y, z) in enumerate(best_conformation):
            color = "red" if self.sequence[i] == "H" else "blue"
            ax.scatter(
                x, y, z, c=color, s=300, alpha=0.8, edgecolors="black", linewidth=2
            )
            ax.text(x + 0.1, y + 0.1, z + 0.1, f"{i}", fontsize=8)

        # Highlight H-H contacts
        contact_pairs: List[Tuple[int, int]] = []
        for i in range(self.length):
            for j in range(i + 2, self.length):
                if self.are_neighbors(best_conformation[i], best_conformation[j]):
                    if self.sequence[i] == "H" and self.sequence[j] == "H":
                        ax.plot(
                            [best_conformation[i, 0], best_conformation[j, 0]],
                            [best_conformation[i, 1], best_conformation[j, 1]],
                            [best_conformation[i, 2], best_conformation[j, 2]],
                            "g--",
                            linewidth=3,
                            alpha=0.7,
                        )
                        contact_pairs.append((i, j))

        ax.set_title(
            f"{title}\nSequence: {self.sequence}\n"
            f"Energy: {energy:.1f} | H-H Contacts: {hh_contacts} | "
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set equal aspect ratio
        coords = best_conformation
        max_range = (
            np.array(
                [
                    coords[:, 0].max() - coords[:, 0].min(),
                    coords[:, 1].max() - coords[:, 1].min(),
                    coords[:, 2].max() - coords[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
        mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
        mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)  # Now recognized

        # Legend
        h_scatter = ax.scatter([], [], [], c="red", s=100, label="H (Hydrophobic)")
        p_scatter = ax.scatter([], [], [], c="blue", s=100, label="P (Polar)")
        if hh_contacts > 0:
            contact_line = plt.Line2D(
                [0],
                [0],
                color="green",
                linestyle="--",
                linewidth=3,
                label="H-H Contact",
            )
            ax.legend(handles=[h_scatter, p_scatter, contact_line])
        else:
            ax.legend(handles=[h_scatter, p_scatter])

        plt.tight_layout()
        plt.show()

        print(f"H-H contact pairs: {contact_pairs}")
        print(f"Total evaluations: {self.evaluation_count}")

    def print_header(self) -> None:
        print(f"Sequence: {self.sequence}")
        print(f"Length: {self.length}")
        print(f"Search space size: 6^{self.length-1} = {6**(self.length-1):.2e}")
