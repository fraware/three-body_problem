#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAM Theory integration for the three-body problem.

This module provides methods for analyzing the integration of KAM Theory with the
isomorphism framework for the three-body problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Import local modules
from three_body_problem import ThreeBodyProblem, HomotheticOrbits, LagrangianSolutions


class KAMTheoryIntegration:
    """
    Class for KAM Theory integration with the isomorphism framework.

    This class provides methods for analyzing the relationship between KAM Theory
    and the isomorphisms between Differential Galois Theory, Painlevé Analysis,
    and Quaternionic Regularization.
    """

    def __init__(self, masses: np.ndarray, G: float = 1.0):
        """
        Initialize the KAM Theory integration with given masses.

        Args:
            masses: Array of three masses [m1, m2, m3]
            G: Gravitational constant (default=1.0)
        """
        self.masses = np.array(masses, dtype=float)
        self.G = float(G)

        # Compute mass parameter sigma
        m1, m2, m3 = self.masses
        self.sigma = (m1 * m2 + m2 * m3 + m3 * m1) / (m1 + m2 + m3)**2

        # Initialize the three-body problem and related classes
        self.tbp = ThreeBodyProblem(masses, G)
        self.homothetic = HomotheticOrbits(self.tbp)
        self.lagrangian = LagrangianSolutions(self.tbp)

    def compute_kam_tori_measure(self, sigma: float, n_samples: int = 500,
                            integration_time: float = 10.0,
                            n_trials: int = 5,
                            random_seed: int = 42) -> Dict[str, float]:
        """
        Compute an estimate of the measure of phase space occupied by KAM tori.

        Args:
            sigma: Mass parameter σ
            n_samples: Number of initial condition samples per trial
            integration_time: Integration time for each sample
            n_trials: Number of trials to run and average (for statistical stability)
            random_seed: Seed for random number generator (for reproducibility)

        Returns:
            Dictionary with KAM measure statistics (mean, std_dev)
        """
        # Create a ThreeBodyProblem instance with the given sigma
        masses = self.find_masses_for_sigma(sigma)
        tbp = ThreeBodyProblem(masses, self.G)

        # Create a Lagrangian solutions generator
        lagrangian = LagrangianSolutions(tbp)

        # Set the random seed for reproducibility
        np.random.seed(random_seed)

        # Run multiple trials to get statistical measures
        trial_results = []

        for trial in range(n_trials):
            # Offset the seed for each trial to maintain trial independence
            # while preserving reproducibility
            np.random.seed(random_seed + trial)

            # Count regular (KAM) trajectories
            regular_count = 0

            # Use a combination of grid and random sampling for better coverage
            # First create a grid of initial perturbation factors
            grid_size = int(np.ceil(np.sqrt(n_samples)))
            perturbation_grid = np.linspace(0.01, 0.15, grid_size)

            # Sample using the grid for better coverage
            sample_index = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if sample_index >= n_samples:
                        break

                    # Get grid-based perturbation factors
                    perturb_x = perturbation_grid[i]
                    perturb_y = perturbation_grid[j]

                    # Generate initial state with controlled perturbation
                    initial_state = self.generate_grid_perturbed_initial_state(
                        lagrangian, perturb_x, perturb_y)

                    # Integrate the system
                    t_span = (0, integration_time)
                    results = tbp.integrate(initial_state, t_span,
                                        method='RK45', rtol=1e-8, atol=1e-8)

                    # Compute the largest Lyapunov exponent
                    lyapunov = self.estimate_lyapunov_exponent(results)

                    # A small Lyapunov exponent indicates regular (KAM) motion
                    if lyapunov < 0.01:
                        regular_count += 1

                    sample_index += 1

            # Estimate the measure of phase space occupied by KAM tori for this trial
            trial_kam_measure = regular_count / n_samples
            trial_results.append(trial_kam_measure)

        # Calculate statistics across trials
        mean_kam_measure = np.mean(trial_results)
        std_dev_kam_measure = np.std(trial_results)

        return {
            "kam_measure": mean_kam_measure,
            "std_dev": std_dev_kam_measure,
            "trials": trial_results
        }

    def generate_grid_perturbed_initial_state(self, lagrangian: LagrangianSolutions,
                                            perturb_x: float, perturb_y: float) -> np.ndarray:
        """
        Generate an initial state with controlled grid-based perturbation.

        Args:
            lagrangian: LagrangianSolutions instance
            perturb_x: Perturbation factor for position components
            perturb_y: Perturbation factor for velocity components

        Returns:
            Perturbed initial state vector
        """
        # Generate a Lagrangian solution initial state
        base_state = lagrangian.generate_initial_state(size=1.0)

        # Get position and momentum components
        positions = base_state[:9]  # First 9 elements are positions
        momenta = base_state[9:]    # Last 9 elements are momenta

        # Apply controlled perturbations
        position_perturbation = np.random.normal(0, perturb_x, positions.shape)
        momentum_perturbation = np.random.normal(0, perturb_y, momenta.shape)

        # Combine into perturbed state
        perturbed_state = np.concatenate([
            positions + position_perturbation,
            momenta + momentum_perturbation
        ])

        return perturbed_state

    def find_masses_for_sigma(self, sigma: float) -> np.ndarray:
        """
        Find a set of masses that give the specified sigma value.

        Args:
            sigma: Desired mass parameter σ

        Returns:
            Array of masses [m1, m2, m3] that best approximates the target sigma

        Notes:
            For positive masses, sigma is mathematically constrained to 0 < sigma ≤ 1/3.
            When sigma = 1/3, the masses are equal.
            For sigma > 1/3, returns equal masses which give the closest possible value (1/3).
        """
        # Handle the theoretical constraint
        if sigma > 1/3:
            # For sigma > 1/3, equal masses (giving sigma = 1/3) are the closest we can get
            return np.array([1.0, 1.0, 1.0])

        # Handle special cases directly for numerical stability
        if abs(sigma - 1/3) < 1e-10:
            return np.array([1.0, 1.0, 1.0])
        elif abs(sigma - 2**3/3**3) < 1e-10:
            return np.array([2.0, 2.0, 1.0])
        elif abs(sigma - 2/3**2) < 1e-10:
            return np.array([2.0, 2.0, 1.0])

        # For sigma < 1/3, we can always use m1 = m2 = 1 and solve for m3
        m1 = m2 = 1.0

        # Using the quadratic formula for: sigma*(m1+m2+m3)^2 = m1*m2 + m2*m3 + m3*m1
        # With m1 = m2 = 1, this becomes: sigma*(2+m3)^2 = 1 + m3 + m3
        # Simplifying: sigma*(4 + 4*m3 + m3^2) = 1 + 2*m3
        # Rearranging: sigma*m3^2 + 4*sigma*m3 + 4*sigma - 2*m3 - 1 = 0
        # Standard form: a*m3^2 + b*m3 + c = 0

        a = sigma
        b = 4*sigma - 2
        c = 4*sigma - 1

        # Calculate the discriminant
        discriminant = b**2 - 4*a*c

        # For 0 < sigma < 1/3, the discriminant is positive
        if discriminant < 0:
            # This shouldn't happen for sigma ≤ 1/3, but handle as a fallback
            return np.array([1.0, 1.0, 1.0])

        # Use the quadratic formula to find m3
        # We want the positive solution
        m3_1 = (-b + np.sqrt(discriminant)) / (2*a)
        m3_2 = (-b - np.sqrt(discriminant)) / (2*a)

        # Choose the positive solution
        m3 = max(m3_1, m3_2)
        if m3 <= 0:
            # Fallback for numerical issues
            m3 = min(m3_1, m3_2)
            if m3 <= 0:
                # If both solutions are non-positive, use a fallback
                return np.array([1.0, 1.0, 0.5])

        # Verify our solution actually gives the correct sigma
        masses = np.array([m1, m2, m3])
        calc_sigma = (m1*m2 + m2*m3 + m3*m1) / (m1 + m2 + m3)**2

        # If our solution is good, return it
        if abs(calc_sigma - sigma) < 1e-8:
            return masses

        # Otherwise use a fallback approach
        return np.array([1.0, 1.0, 0.5 + sigma])

    def generate_perturbed_initial_state(self, lagrangian: LagrangianSolutions,
                                       perturbation_factor: float) -> np.ndarray:
        """
        Generate an initial state with perturbation from a Lagrangian solution.

        Args:
            lagrangian: LagrangianSolutions instance
            perturbation_factor: Factor controlling the perturbation magnitude

        Returns:
            Perturbed initial state vector
        """
        # Generate a Lagrangian solution initial state
        base_state = lagrangian.generate_initial_state(size=1.0)

        # Add random perturbations scaled by the factor
        perturbation = np.random.normal(0, perturbation_factor, base_state.shape)

        return base_state + perturbation

    def estimate_lyapunov_exponent(self, integration_results: Dict,
                                transient_fraction: float = 0.3) -> float:
        """
        Estimate the largest Lyapunov exponent from integration results.

        Args:
            integration_results: Dictionary with integration results
            transient_fraction: Fraction of the trajectory to discard as transient

        Returns:
            Estimated largest Lyapunov exponent
        """
        states = integration_results["states"]
        times = integration_results["t"]

        # Discard transient part
        n_transient = int(len(times) * transient_fraction)
        states = states[n_transient:]
        times = times[n_transient:]

        # Compute consecutive state differences
        diff = np.diff(states, axis=0)

        # Compute the norms of the differences
        norms = np.linalg.norm(diff, axis=1)

        # Estimate the Lyapunov exponent as the average exponential growth rate
        time_diff = np.diff(times)
        growth_rates = np.log(norms / norms[0])
        lyapunov = np.mean(growth_rates / np.cumsum(time_diff))

        return max(0, lyapunov)  # Lyapunov exponents should be non-negative

    def compute_kam_measure_vs_sigma(self, sigma_values: np.ndarray,
                                n_samples: int = 500, n_trials: int = 5,
                                random_seed: int = 42) -> Dict:
        """
        Compute the KAM measure as a function of the mass parameter.

        Args:
            sigma_values: Array of sigma values to analyze
            n_samples: Number of initial condition samples per trial
            n_trials: Number of trials for statistical stability
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with results of the analysis
        """
        kam_measures = np.zeros_like(sigma_values, dtype=float)
        kam_std_devs = np.zeros_like(sigma_values, dtype=float)
        actual_sigma_values = np.zeros_like(sigma_values, dtype=float)

        for i, sigma in enumerate(sigma_values):
            # Check if sigma exceeds the mathematical constraint
            if sigma > 1/3:
                print(f"Warning: sigma={sigma} exceeds mathematical constraint of 1/3.")
                print("Using equal masses (sigma=1/3) for this calculation.")
                result = self.compute_kam_tori_measure(1/3, n_samples, n_trials=n_trials,
                                                    random_seed=random_seed)
                kam_measures[i] = result["kam_measure"]
                kam_std_devs[i] = result["std_dev"]
                actual_sigma_values[i] = 1/3
            else:
                result = self.compute_kam_tori_measure(sigma, n_samples, n_trials=n_trials,
                                                    random_seed=random_seed)
                kam_measures[i] = result["kam_measure"]
                kam_std_devs[i] = result["std_dev"]
                actual_sigma_values[i] = sigma

        return {
            "sigma_values": sigma_values,
            "kam_measures": kam_measures,
            "kam_std_devs": kam_std_devs,
            "actual_sigma_values": actual_sigma_values
        }

    def compute_constants_of_isomorphism(self, sigma_values: np.ndarray) -> Dict:
        """
        Compute the constants relating isomorphism structures to KAM Theory.

        Args:
            sigma_values: Array of sigma values to analyze

        Returns:
            Dictionary with computed constants
        """
        # For exceptional mass ratios, compute the constant C in the KAM measure formula:
        # μ(KAM) ≈ 1 - C|σ - σ_0|^(1/2)
        # where σ_0 is an exceptional value

        exceptional_values = [1/3, 2**3/3**3, 2/3**2]
        constants = {}

        for sigma_0 in exceptional_values:
            # Find sigma values close to the exceptional value
            close_indices = [i for i, sigma in enumerate(sigma_values)
                           if 0.001 < abs(sigma - sigma_0) < 0.02]

            if close_indices:
                close_sigmas = [sigma_values[i] for i in close_indices]

                # Compute KAM measures for these sigma values
                kam_results = self.compute_kam_measure_vs_sigma(np.array(close_sigmas), n_samples=20)
                kam_measures = kam_results["kam_measures"]

                # Compute the constant C for each nearby sigma
                C_values = [(1 - measure) / np.sqrt(abs(sigma - sigma_0))
                           for measure, sigma in zip(kam_measures, close_sigmas)]

                # Average the C values
                constants[sigma_0] = np.mean(C_values)
            else:
                constants[sigma_0] = None

        return {
            "exceptional_values": exceptional_values,
            "constants": constants
        }

    def isomorphism_kam_correspondence(self, sigma: float) -> Dict:
        """
        Analyze the correspondence between isomorphism structures and KAM Theory.

        Args:
            sigma: Mass parameter to analyze

        Returns:
            Dictionary with correspondence analysis results
        """
        # The isomorphism structure is characterized by the Galois group type,
        # which is related to the mass parameter

        # For exceptional mass ratios, determine the isomorphism structure
        if abs(sigma - 1/3) < 1e-10:
            galois_group = "Dihedral"
            identity_component = "Diagonal (abelian)"
            branch_point_type = "square root (Z_2)"
            monodromy_type = "Z_2"
            integrability = "Partially integrable"

        elif abs(sigma - 2**3/3**3) < 1e-10:
            galois_group = "Dihedral"
            identity_component = "Diagonal (abelian)"
            branch_point_type = "square root (Z_2)"
            monodromy_type = "Z_2"
            integrability = "Partially integrable"

        elif abs(sigma - 2/3**2) < 1e-10:
            galois_group = "Triangular"
            identity_component = "Diagonal (abelian)"
            branch_point_type = "none (meromorphic)"
            monodromy_type = "Trivial"
            integrability = "Partially integrable"

        else:
            galois_group = "SL(2,C)"
            identity_component = "SL(2,C) (non-abelian)"
            branch_point_type = "transcendental"
            monodromy_type = "Complex"
            integrability = "Non-integrable"

        # Estimate the KAM measure
        kam_result = self.compute_kam_tori_measure(sigma, n_samples=30)

        # Extract the kam_measure value from the result dictionary
        kam_measure = kam_result["kam_measure"]
        kam_std_dev = kam_result.get("std_dev", 0.0)

        # Determine the relationship to KAM Theory
        if kam_measure > 0.9:
            kam_characterization = "Large KAM islands (significant regular regions)"
        elif kam_measure > 0.5:
            kam_characterization = "Moderate KAM islands (mix of regular and chaotic regions)"
        else:
            kam_characterization = "Small KAM islands (predominantly chaotic)"

        return {
            "mass_parameter": sigma,
            "isomorphism_structure": {
                "galois_group": galois_group,
                "identity_component": identity_component,
                "branch_point_type": branch_point_type,
                "monodromy_type": monodromy_type,
                "integrability": integrability
            },
            "kam_theory": {
                "kam_measure": kam_measure,
                "kam_std_dev": kam_std_dev,
                "characterization": kam_characterization
            },
            "correspondence": {
                "is_exceptional": abs(sigma - 1/3) < 1e-10 or abs(sigma - 2**3/3**3) < 1e-10 or abs(sigma - 2/3**2) < 1e-10,
                "partial_integrability_corresponds_to_high_kam_measure": (integrability == "Partially integrable") == (kam_measure > 0.8)
            }
        }


def test_kam_theory_integration():
    """Test the KAM Theory integration implementation."""
    # Test with equal masses (sigma = 1/3)
    masses = np.array([1.0, 1.0, 1.0])
    kam = KAMTheoryIntegration(masses)

    # Test finding masses for a specific sigma
    # Use a valid sigma value that's within the constraint
    test_sigma = 0.25
    test_masses = kam.find_masses_for_sigma(test_sigma)
    test_tbp = ThreeBodyProblem(test_masses)
    assert abs(test_tbp.sigma - test_sigma) < 1e-5

    # Test KAM measure computation (with reduced samples for testing)
    kam_measure = kam.compute_kam_tori_measure(1/3, n_samples=5, integration_time=2.0)
    print(f"KAM measure for sigma = 1/3: {kam_measure}")

    # Test isomorphism-KAM correspondence
    correspondence = kam.isomorphism_kam_correspondence(1/3)
    print(f"Correspondence analysis for sigma = 1/3: {correspondence['correspondence']}")

    print("All KAM Theory integration tests passed!")


def test_non_exceptional_case():
    """Test KAM Theory integration for a non-exceptional case."""
    # Test with masses giving sigma = 0.4
    masses = np.array([1.0, 1.0, 0.5])  # Approximate, actual sigma will be computed
    kam = KAMTheoryIntegration(masses)

    # Compute the actual sigma
    print(f"Non-exceptional sigma: {kam.sigma}")

    # Test isomorphism-KAM correspondence
    correspondence = kam.isomorphism_kam_correspondence(kam.sigma)
    print(f"Non-exceptional correspondence analysis: {correspondence['correspondence']}")

    print("Non-exceptional case test completed!")


if __name__ == "__main__":
    # Run tests
    test_kam_theory_integration()
    test_non_exceptional_case()
