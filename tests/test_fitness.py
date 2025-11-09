import numpy as np
import pytest

from src.pfo_base import PFOBase

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_model():
    """Fixture for a simple HP model"""
    return PFOBase("HPHPH")


@pytest.fixture
def collision_moves():
    """Fixture for moves that cause collision"""
    return np.array([0, 1, 0, 0])


# ============================================================================
# Tests for PFOBase - Fitness Function
# ============================================================================


class TestFitnessFunction:
    """Test fitness function"""

    def test_fitness_function_valid(self, simple_model):
        """Test fitness function with valid moves"""
        moves = np.array([0.3, 2.7, 4.1, 1.8])

        initial_eval_count = simple_model.evaluation_count
        fitness = simple_model.fitness_function(moves)

        assert simple_model.evaluation_count == initial_eval_count + 1
        assert isinstance(fitness, int)
        assert np.isfinite(fitness)

    def test_fitness_function_invalid(self, simple_model, collision_moves):
        """Test fitness function with invalid (colliding) moves"""
        fitness = simple_model.fitness_function(collision_moves)
        assert fitness == 0

    def test_fitness_function_rounding(self, simple_model):
        """Test that continuous values are properly rounded"""
        moves_float = np.array([0.4, 2.6, 4.2, 1.7])
        moves_int = np.array([0, 3, 4, 2])

        fitness1 = simple_model.fitness_function(moves_float)

        # Reset best to test again
        simple_model.best_energy = float("inf")
        fitness2 = simple_model.fitness_function(moves_int)

        assert isinstance(fitness1, int)
        assert isinstance(fitness2, int)
        assert fitness1 == fitness2

    def test_fitness_function_clipping(self, simple_model):
        """Test that out-of-bounds values are clipped"""
        moves = np.array([-1, 6, 10, -5])  # All out of bounds

        fitness = simple_model.fitness_function(moves)

        assert isinstance(fitness, int)
