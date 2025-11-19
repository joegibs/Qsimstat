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
import time
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


def run_circuit_with_timing(exp, instruction, state, depth, iter_idx, timing_dict):
    """
    Wrapper around run_circuit that tracks timing for gates vs data computation.

    Args:
        exp: Experiment instance
        instruction: Circuit instruction string
        state: MPS state
        depth: Circuit depth
        iter_idx: Iteration index
        timing_dict: Dictionary to store timing results
    """
    from gates import Sample_Clifford, PPgate

    gate_time = 0
    data_time = 0
    data_breakdown = {stat: 0 for stat in exp.statstag.keys() if exp.statstag[stat]}

    if instruction == "random_matchgate_brickwork":
        for d in range(depth//2):
            # Time gate applications
            gate_start = time.time()
            # Odd bonds
            for site in range(0, exp.N-1, 2):
                state.gate_split(PPgate(), (site, site+1), inplace=True)
            # Even bonds
            for site in range(1, exp.N-1, 2):
                state.gate_split(PPgate(), (site, site+1), inplace=True)
            gate_time += time.time() - gate_start

            # Time data collection
            if exp.circuittag["dataC"] != 'end':
                if d*2 % exp.circuittag["dataC"] == 0 and d*2 != depth:
                    point = d*2 // exp.circuittag["dataC"]
                    data_start = time.time()
                    compute_data_with_timing(exp, state, iter_idx, point, data_breakdown)
                    data_time += time.time() - data_start

    elif instruction == "random_clifford_brickwork":
        for d in range(depth//2):
            # Time gate applications
            gate_start = time.time()
            # Odd bonds
            for site in range(0, exp.N-1, 2):
                state.gate_split(Sample_Clifford(), (site, site+1), inplace=True)
            # Even bonds
            for site in range(1, exp.N-1, 2):
                state.gate_split(Sample_Clifford(), (site, site+1), inplace=True)
            gate_time += time.time() - gate_start

            # Time data collection
            if exp.circuittag["dataC"] != 'end':
                if d*2 % exp.circuittag["dataC"] == 0 and d*2 != depth:
                    point = d*2 // exp.circuittag["dataC"]
                    data_start = time.time()
                    compute_data_with_timing(exp, state, iter_idx, point, data_breakdown)
                    data_time += time.time() - data_start

    timing_dict['gate_time'] = gate_time
    timing_dict['data_time'] = data_time
    timing_dict['data_breakdown'] = data_breakdown


def compute_data_with_timing(exp, state, it, point, data_breakdown):
    """Compute data with per-statistic timing."""
    from DataFunctions import Renyi10, FAF, ESS, flatness, IPR, bonddim, repulsion, SFF, porter_thomas, SRE

    if exp.statstag["renyi10"]:
        start = time.time()
        exp.frames["renyi10"][it, point] = Renyi10(state, exp.N)
        data_breakdown["renyi10"] += time.time() - start

    if exp.statstag["faf"]:
        start = time.time()
        faf_values = np.zeros(10)
        for k in range(1, 11):
            faf_values[k-1] = FAF(state, exp.encoding, exp.N, k)
        exp.frames["faf"][it, point] = faf_values
        data_breakdown["faf"] += time.time() - start

    if exp.statstag["ipr"]:
        start = time.time()
        exp.frames["ipr"][it, point] = IPR(state)
        data_breakdown["ipr"] += time.time() - start

    if exp.statstag["ess"]:
        start = time.time()
        exp.frames["ess"][it, point] = ESS(state, exp.N)
        data_breakdown["ess"] += time.time() - start

    if exp.statstag["bonddim"]:
        start = time.time()
        exp.frames["bonddim"][it, point] = bonddim(state)
        data_breakdown["bonddim"] += time.time() - start

    if exp.statstag["flatness"]:
        start = time.time()
        exp.frames["flatness"][it, point] = flatness(state, exp.N)
        data_breakdown["flatness"] += time.time() - start

    if exp.statstag["repulsion"]:
        start = time.time()
        exp.frames["repulsion"][it, point] = repulsion(state, exp.N)
        data_breakdown["repulsion"] += time.time() - start

    if exp.statstag["sff"]:
        start = time.time()
        exp.frames["sff"][it, point] = SFF(state, exp.N)
        data_breakdown["sff"] += time.time() - start

    if exp.statstag["porter_thomas"]:
        start = time.time()
        exp.frames["porter_thomas"][it, point] = porter_thomas(state, exp.N)
        data_breakdown["porter_thomas"] += time.time() - start

    if exp.statstag["sre"]:
        start = time.time()
        exp.frames["sre"][it, point] = SRE(state, exp.N)
        data_breakdown["sre"] += time.time() - start


def run_experiment_with_initial_mps(exp):
    """
    Run the experiment using Experiment class methods with detailed timing.

    This initializes with random MPS and uses custom timing-enabled circuit runners
    for both matchgate and Clifford phases.
    """
    print("\nRunning experiment...")

    # Initialize timing trackers
    all_gate_times = []
    all_data_times = []
    data_breakdown_totals = {stat: [] for stat in exp.statstag.keys() if exp.statstag[stat]}

    for it in range(exp.circuittag["iter"]):
        print(f"\nIteration {it+1}/{exp.circuittag['iter']}")
        iter_start = time.time()

        # Initialize random MPS with entanglement
        init_start = time.time()
        state = qtn.MPS_computational_state('0' * exp.N)
        for site in range(exp.N-1):
            state.gate_split(make_ortho(4, 0, 1), (site, site+1), inplace=True)
        init_time = time.time() - init_start

        # Phase 1: Matchgate circuit with timing
        timing_matchgate = {}
        run_circuit_with_timing(exp, "random_matchgate_brickwork", state, matchgate_depth, it, timing_matchgate)

        # Phase 2: Clifford circuit with timing
        timing_clifford = {}
        run_circuit_with_timing(exp, "random_clifford_brickwork", state, clifford_depth, it, timing_clifford)

        # Aggregate timing
        total_gate_time = timing_matchgate['gate_time'] + timing_clifford['gate_time']
        total_data_time = timing_matchgate['data_time'] + timing_clifford['data_time']

        all_gate_times.append(total_gate_time)
        all_data_times.append(total_data_time)

        # Aggregate data breakdown
        for stat in data_breakdown_totals.keys():
            stat_time = timing_matchgate['data_breakdown'].get(stat, 0) + timing_clifford['data_breakdown'].get(stat, 0)
            data_breakdown_totals[stat].append(stat_time)

        iter_total = time.time() - iter_start

        print(f"  Iteration time: {iter_total:.2f}s")
        print(f"    Initialization: {init_time:.3f}s")
        print(f"    Gate evolution: {total_gate_time:.3f}s")
        print(f"    Data collection: {total_data_time:.3f}s")

    # Print timing summary
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"\nPer iteration averages ({exp.circuittag['iter']} iterations):")
    print(f"  Gate evolution:   {np.mean(all_gate_times):.3f}s ± {np.std(all_gate_times):.3f}s")
    print(f"  Data collection:  {np.mean(all_data_times):.3f}s ± {np.std(all_data_times):.3f}s")
    print(f"  Total per iter:   {np.mean(all_gate_times) + np.mean(all_data_times):.3f}s")

    print(f"\nData collection breakdown (average per iteration):")
    for stat, times in sorted(data_breakdown_totals.items(), key=lambda x: np.mean(x[1]), reverse=True):
        avg_time = np.mean(times)
        if avg_time > 0.001:  # Only show if > 1ms
            print(f"  {stat:15s}: {avg_time:.3f}s ± {np.std(times):.3f}s")

    total_time = sum(all_gate_times) + sum(all_data_times)
    print(f"\nTotal experiment time: {total_time:.2f}s")
    print(f"  Gate evolution:   {sum(all_gate_times):.2f}s ({100*sum(all_gate_times)/total_time:.1f}%)")
    print(f"  Data collection:  {sum(all_data_times):.2f}s ({100*sum(all_data_times)/total_time:.1f}%)")

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
