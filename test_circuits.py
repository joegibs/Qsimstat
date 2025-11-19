"""
Test script for quantum circuit evolution using the Experiment class.

Tests two circuit types (Clifford and Matchgate) on three initial states:
1. Vacuum state: |00000000>
2. Product state: random single-qubit rotations
3. Random MPS: entangled initial state

Computes Fermionic Anti-Flatness (FAF), Renyi entropies, and other statistics
every 10 layers with 1 iteration to verify the Experiment class works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from Experiment import Experiment
from gates import JW
import quimb.tensor as qtn
from gates import make_ortho

# Test parameters
N = 8  # Number of qubits
depth = 100  # Total circuit depth
data_interval = 10  # Compute data every 10 layers
iterations = 1  # Single iteration for testing


def initialize_vacuum_state(N):
    """Initialize vacuum state |00000000>."""
    return qtn.MPS_computational_state('0' * N)


def initialize_product_state(N):
    """Initialize random product state with single-qubit rotations."""
    state = qtn.MPS_computational_state('0' * N)
    for site in range(N):
        state.gate(make_ortho(2, 0, 1), site, inplace=True, contract=True)
    return state


def initialize_random_mps(N):
    """Initialize random MPS with entanglement."""
    state = qtn.MPS_computational_state('0' * N)
    # Apply random two-qubit gates to create entanglement
    for site in range(N-1):
        state.gate_split(make_ortho(4, 0, 1), (site, site+1), inplace=True)
    return state


def setup_experiment(circuit_type, initial_state_name):
    """Set up experiment for given circuit type and initial state.

    Args:
        circuit_type: 'clifford' or 'matchgate'
        initial_state_name: 'vacuum', 'product', or 'random_mps'
    """
    exp = Experiment(N)

    # Configure tracking
    exp.tracktag["mps"] = True

    # Configure statistics to compute
    exp.statstag["renyi10"] = True
    exp.statstag["faf"] = True
    exp.statstag["ess"] = True
    exp.statstag["flatness"] = True
    exp.statstag["ipr"] = True

    # Configure circuit
    exp.circuittag["depth"] = depth
    exp.circuittag["iter"] = iterations
    exp.circuittag["dataC"] = data_interval

    if circuit_type == 'clifford':
        exp.circuittag["sequence"] = ["random_clifford_brickwork"]
    elif circuit_type == 'matchgate':
        exp.circuittag["sequence"] = ["random_matchgate_brickwork"]

    # Set up encoding for FAF
    exp.encoding = JW(N)

    # Initialize data frames
    # We collect at layers 10, 20, ..., 100 (skipping layer 0), so num_data_points = depth // data_interval
    # But since we skip 0, we actually get depth // data_interval points
    num_data_points = depth // data_interval
    exp.frames["renyi10"] = np.zeros((iterations, num_data_points, 10))
    exp.frames["faf"] = np.zeros((iterations, num_data_points, 10))
    exp.frames["ess"] = np.zeros((iterations, num_data_points))
    exp.frames["flatness"] = np.zeros((iterations, num_data_points))
    exp.frames["ipr"] = np.zeros((iterations, num_data_points))

    # Store metadata
    exp.circuit_type = circuit_type
    exp.initial_state_name = initial_state_name

    return exp


def run_experiment(exp):
    """Run the experiment with the specified circuit and initial state."""
    from gates import Sample_Clifford, PPgate

    print(f"\n{'='*60}")
    print(f"Running: {exp.circuit_type.upper()} on {exp.initial_state_name.upper()} state")
    print(f"{'='*60}")

    # Initialize state based on type
    if exp.initial_state_name == 'vacuum':
        state = initialize_vacuum_state(exp.N)
    elif exp.initial_state_name == 'product':
        state = initialize_product_state(exp.N)
    elif exp.initial_state_name == 'random_mps':
        state = initialize_random_mps(exp.N)

    print(f"  Initial state: {exp.initial_state_name}")
    print(f"  Circuit type: {exp.circuit_type}")

    # Run circuit
    iter_idx = 0  # Single iteration
    instruction = exp.circuittag["sequence"][0]

    run_circuit_fixed(exp, instruction, state, exp.circuittag["depth"], iter_idx)

    print("\n  Experiment complete!")

    # Save data to disk
    print("  Saving data...")
    exp.save_data()

    return exp


def run_circuit_fixed(exp, instruction, state, depth, iter_idx):
    """Fixed version of run_circuit that properly handles brickwork patterns."""
    from gates import Sample_Clifford, PPgate

    data_interval = int(exp.circuittag["dataC"])

    if instruction == "random_clifford_brickwork":
        for d in range(depth // 2):
            # Odd bonds
            for site in range(0, exp.N-1, 2):
                state.gate_split(Sample_Clifford(), (site, site+1), inplace=True)
            # Even bonds
            for site in range(1, exp.N-1, 2):
                state.gate_split(Sample_Clifford(), (site, site+1), inplace=True)

            # Collect data
            layer = (d + 1) * 2  # Current layer after this brickwork step
            if layer % data_interval == 0:
                point = layer // data_interval - 1
                print(f"    Layer {layer}: collecting data (point {point})")
                compute_data_fixed(exp, state, iter_idx, point)

    elif instruction == "random_matchgate_brickwork":
        for d in range(depth // 2):
            # Odd bonds
            for site in range(0, exp.N-1, 2):
                state.gate_split(PPgate(), (site, site+1), inplace=True)
            # Even bonds
            for site in range(1, exp.N-1, 2):
                state.gate_split(PPgate(), (site, site+1), inplace=True)

            # Collect data
            layer = (d + 1) * 2  # Current layer after this brickwork step
            if layer % data_interval == 0:
                point = layer // data_interval - 1
                print(f"    Layer {layer}: collecting data (point {point})")
                compute_data_fixed(exp, state, iter_idx, point)


def compute_data_fixed(exp, state, it, point):
    """Fixed version of compute_data that properly computes all statistics."""
    from DataFunctions import Renyi10, FAF, ESS, flatness, IPR

    if exp.statstag["renyi10"]:
        exp.frames["renyi10"][it, point] = Renyi10(state, exp.N)

    if exp.statstag["faf"]:
        faf_values = np.zeros(10)
        for k in range(1, 11):
            faf_values[k-1] = FAF(state, exp.encoding, exp.N, k)
        exp.frames["faf"][it, point] = faf_values

    if exp.statstag["ess"]:
        exp.frames["ess"][it, point] = ESS(state, exp.N)

    if exp.statstag["flatness"]:
        exp.frames["flatness"][it, point] = flatness(state, exp.N)

    if exp.statstag["ipr"]:
        exp.frames["ipr"][it, point] = IPR(state)


def plot_all_results(results_dict):
    """Plot comparison of all circuit types and initial states.

    Args:
        results_dict: Dictionary with keys like 'clifford_vacuum', 'matchgate_product', etc.
    """
    num_points = results_dict[list(results_dict.keys())[0]].frames["renyi10"].shape[1]
    layers = np.arange(1, num_points + 1) * data_interval

    # Create a large figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f'Circuit Comparison: {N} qubits, {depth} layers', fontsize=16)

    colors = {'vacuum': 'blue', 'product': 'green', 'random_mps': 'red'}
    markers = {'clifford': 'o', 'matchgate': 's'}

    # Von Neumann Entropy - Clifford
    ax = axes[0, 0]
    for init_state in ['vacuum', 'product', 'random_mps']:
        key = f'clifford_{init_state}'
        if key in results_dict:
            exp = results_dict[key]
            ax.plot(layers, exp.frames["renyi10"][0, :, 1],
                   marker=markers['clifford'], color=colors[init_state],
                   label=init_state, linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Von Neumann Entropy')
    ax.set_title('Clifford: Entanglement Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Von Neumann Entropy - Matchgate
    ax = axes[0, 1]
    for init_state in ['vacuum', 'product', 'random_mps']:
        key = f'matchgate_{init_state}'
        if key in results_dict:
            exp = results_dict[key]
            ax.plot(layers, exp.frames["renyi10"][0, :, 1],
                   marker=markers['matchgate'], color=colors[init_state],
                   label=init_state, linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Von Neumann Entropy')
    ax.set_title('Matchgate: Entanglement Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FAF (k=2) - Clifford
    ax = axes[1, 0]
    for init_state in ['vacuum', 'product', 'random_mps']:
        key = f'clifford_{init_state}'
        if key in results_dict:
            exp = results_dict[key]
            ax.plot(layers, exp.frames["faf"][0, :, 1],
                   marker=markers['clifford'], color=colors[init_state],
                   label=init_state, linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('FAF (k=2)')
    ax.set_title('Clifford: Fermionic Anti-Flatness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FAF (k=2) - Matchgate
    ax = axes[1, 1]
    for init_state in ['vacuum', 'product', 'random_mps']:
        key = f'matchgate_{init_state}'
        if key in results_dict:
            exp = results_dict[key]
            ax.plot(layers, exp.frames["faf"][0, :, 1],
                   marker=markers['matchgate'], color=colors[init_state],
                   label=init_state, linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('FAF (k=2)')
    ax.set_title('Matchgate: Fermionic Anti-Flatness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IPR - Clifford (log scale)
    ax = axes[2, 0]
    for init_state in ['vacuum', 'product', 'random_mps']:
        key = f'clifford_{init_state}'
        if key in results_dict:
            exp = results_dict[key]
            ax.semilogy(layers, exp.frames["ipr"][0, :],
                       marker=markers['clifford'], color=colors[init_state],
                       label=init_state, linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('IPR (log scale)')
    ax.set_title('Clifford: Inverse Participation Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IPR - Matchgate (log scale)
    ax = axes[2, 1]
    for init_state in ['vacuum', 'product', 'random_mps']:
        key = f'matchgate_{init_state}'
        if key in results_dict:
            exp = results_dict[key]
            ax.semilogy(layers, exp.frames["ipr"][0, :],
                       marker=markers['matchgate'], color=colors[init_state],
                       label=init_state, linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('IPR (log scale)')
    ax.set_title('Matchgate: Inverse Participation Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_circuits_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'test_circuits_comparison.png'")
    plt.show()


def main():
    """Run all tests: 2 circuit types x 3 initial states = 6 tests."""
    print(f"\nTest Configuration:")
    print(f"  Number of qubits (N): {N}")
    print(f"  Circuit depth: {depth}")
    print(f"  Data collection interval: {data_interval} layers")
    print(f"  Iterations: {iterations}")
    print(f"\nRunning 6 tests (2 circuit types x 3 initial states)...\n")

    results = {}

    # Test all combinations
    for circuit_type in ['clifford', 'matchgate']:
        for init_state in ['vacuum', 'product', 'random_mps']:
            key = f'{circuit_type}_{init_state}'

            exp = setup_experiment(circuit_type, init_state)
            exp = run_experiment(exp)
            results[key] = exp

    # Plot comparison
    print("\n" + "="*60)
    print("Generating comparison plots...")
    print("="*60)
    plot_all_results(results)

    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)

    print("\nExpected behaviors:")
    print("  Clifford circuits:")
    print("    - Should thermalize (high entanglement entropy)")
    print("    - FAF should remain low (Cliffords are Gaussian)")
    print("    - IPR should decrease (delocalization)")
    print("\n  Matchgate circuits:")
    print("    - Should be integrable (moderate entanglement growth)")
    print("    - FAF should remain low (matchgates are free fermions)")
    print("    - May show different IPR behavior")
    print("\n  Initial state dependence:")
    print("    - Vacuum: starts with zero entanglement")
    print("    - Product: starts with zero entanglement")
    print("    - Random MPS: starts with some entanglement")


if __name__ == "__main__":
    main()
