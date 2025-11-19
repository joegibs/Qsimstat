"""
Experiment: Clifford circuits acting after matchgates on random MPS.

This experiment studies the behavior of random Clifford circuits applied after
random matchgate circuits on random matrix product states.

Physical Motivation:
- Matchgates are free fermion (integrable) gates that preserve Gaussianity
- Cliffords are also Gaussian gates
- Starting from a random MPS breaks integrability
- This setup explores thermalization and entanglement dynamics

Data Collected:
- Renyi entropies (k=0 to 9)
- Fermionic Anti-Flatness (FAF) for k=1 to 10
- Inverse Participation Ratio (IPR)
- Entanglement Spectrum Statistics (ESS)
- Bond dimensions
- Flatness of entanglement spectrum
- Level repulsion statistics
- Spectral Form Factor (SFF)
- Porter-Thomas distribution analysis
- Stabilizer Renyi Entropy (SRE)

Circuit Structure:
1. Initialize random MPS (entangled initial state)
2. Apply random matchgate brickwork circuit (depth: matchgate_depth)
3. Apply random Clifford brickwork circuit (depth: clifford_depth)
4. Collect data every data_interval layers
"""

import sys
import os
# Add parent directory to path to import Experiment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Experiment import Experiment
from gates import JW, make_ortho
import numpy as np
import quimb.tensor as qtn

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# System size
N = 10  # Number of qubits

# Circuit depths
matchgate_depth = 50   # Depth of matchgate circuit
clifford_depth = 50    # Depth of Clifford circuit after matchgates
total_depth = matchgate_depth + clifford_depth

# Data collection
data_interval = 10  # Collect data every 10 layers
iterations = 10     # Number of independent runs for statistics

# Results directory
results_dir = "results/clifford_after_matchgate"

# ============================================================================
# EXPERIMENT SETUP
# ============================================================================

def setup_experiment():
    """Set up the experiment with all data metrics enabled."""

    print("="*70)
    print("EXPERIMENT: Clifford After Matchgate on Random MPS")
    print("="*70)
    print(f"\nParameters:")
    print(f"  System size (N): {N} qubits")
    print(f"  Matchgate depth: {matchgate_depth} layers")
    print(f"  Clifford depth: {clifford_depth} layers")
    print(f"  Total depth: {total_depth} layers")
    print(f"  Data interval: every {data_interval} layers")
    print(f"  Iterations: {iterations}")
    print(f"  Results directory: {results_dir}")

    # Initialize experiment
    exp = Experiment(N)

    # Enable MPS tracking
    exp.tracktag["mps"] = True

    # Enable ALL statistics
    exp.statstag["renyi10"] = True
    exp.statstag["faf"] = True
    exp.statstag["ipr"] = True
    exp.statstag["ess"] = True
    exp.statstag["bonddim"] = True
    exp.statstag["flatness"] = True
    exp.statstag["repulsion"] = True
    exp.statstag["sff"] = True
    exp.statstag["porter_thomas"] = True
    exp.statstag["sre"] = True

    # Configure circuit - note that we use a single combined depth
    exp.circuittag["depth"] = matchgate_depth + clifford_depth
    exp.circuittag["iter"] = iterations
    exp.circuittag["sequence"] = ["random_matchgate_brickwork", "random_clifford_brickwork"]
    exp.circuittag["dataC"] = data_interval

    # Set up Jordan-Wigner encoding for FAF calculation
    exp.encoding = JW(N)

    # Initialize data frames
    # Since we have TWO circuit types, total data points come from BOTH phases
    num_data_points = (matchgate_depth + clifford_depth) // data_interval
    exp.frames["renyi10"] = np.zeros((iterations, num_data_points, 10))
    exp.frames["faf"] = np.zeros((iterations, num_data_points, 10))
    exp.frames["ipr"] = np.zeros((iterations, num_data_points))
    exp.frames["ess"] = np.zeros((iterations, num_data_points))
    exp.frames["bonddim"] = np.zeros((iterations, num_data_points, N-1))
    exp.frames["flatness"] = np.zeros((iterations, num_data_points))
    exp.frames["repulsion"] = np.zeros((iterations, num_data_points, 200))
    exp.frames["sff"] = np.zeros((iterations, num_data_points))
    exp.frames["porter_thomas"] = np.empty((iterations, num_data_points), dtype=object)
    exp.frames["sre"] = np.zeros((iterations, num_data_points))

    print("\nAll statistics enabled:")
    for stat, enabled in exp.statstag.items():
        if enabled:
            print(f"  ✓ {stat}")

    return exp


def run_experiment_with_initial_mps(exp):
    """
    Run the experiment using Experiment class methods.

    This initializes with random MPS and uses the run_circuit() method
    from the Experiment class for both matchgate and Clifford phases.
    """
    print("\nRunning experiment...")

    for it in range(exp.circuittag["iter"]):
        print(f"Iteration {it+1}/{exp.circuittag['iter']}")

        # Initialize random MPS with entanglement
        state = qtn.MPS_computational_state('0' * exp.N)
        for site in range(exp.N-1):
            state.gate_split(make_ortho(4, 0, 1), (site, site+1), inplace=True)

        # Phase 1: Matchgate circuit - use Experiment.run_circuit()
        exp.run_circuit("random_matchgate_brickwork", state, matchgate_depth, it)

        # Phase 2: Clifford circuit - use Experiment.run_circuit()
        exp.run_circuit("random_clifford_brickwork", state, clifford_depth, it)

    return exp


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Setup experiment
    exp = setup_experiment()

    # Run experiment
    exp = run_experiment_with_initial_mps(exp)

    # Save data
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    exp.save_data(base_dir=results_dir)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}/")

if __name__ == "__main__":
    main()
