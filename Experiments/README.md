# Experiments

This directory contains experimental scripts for studying quantum circuit dynamics using the Qsimstat framework.

## Directory Structure

```
Experiments/
├── README.md                        # This file
└── clifford_after_matchgate.py     # Clifford-after-matchgate experiment
```

## Available Experiments

### 1. `clifford_after_matchgate.py`

**Purpose:** Study the transition from integrable (matchgate) to chaotic (Clifford) dynamics.

**Circuit Structure:**
1. Random MPS initialization (entangled initial state)
2. Random matchgate circuit (depth: 50 layers)
3. Random Clifford circuit (depth: 50 layers)

**Parameters:**
- System size: 10 qubits (configurable)
- Matchgate depth: 50 layers
- Clifford depth: 50 layers
- Data collected every 10 layers
- 10 iterations for statistics

**Data Collected:**
- Renyi entropies (k=0 to 9)
- Fermionic Anti-Flatness (FAF, k=1 to 10)
- Inverse Participation Ratio (IPR)
- Entanglement Spectrum Statistics (ESS)
- Bond dimensions
- Entanglement spectrum flatness
- Level repulsion distribution

**Usage:**
```bash
cd Experiments
python clifford_after_matchgate.py
```

**Output:**
Results saved to `results/clifford_after_matchgate/` with subdirectories for each metric.

**Expected Results:**
- Matchgate phase: Integrable dynamics, FAF ≈ 0
- Clifford phase: Thermalization, continued FAF ≈ 0 (both are Gaussian)
- Transition at layer 50: Possible change in entanglement growth rate

## Creating New Experiments

To create a new experiment:

1. **Copy the template:**
   ```bash
   cp clifford_after_matchgate.py my_new_experiment.py
   ```

2. **Modify parameters:**
   - Adjust `N`, `depth`, `data_interval`, `iterations`
   - Change `results_dir` to a unique name
   - Update circuit sequence in `exp.circuittag["sequence"]`

3. **Customize circuit evolution:**
   - Modify `run_experiment_custom()` to implement your circuit
   - Update initial state preparation as needed

4. **Document your experiment:**
   - Update the docstring at the top of the file
   - Add entry to this README

## Tips

- **Start small:** Test with small N (e.g., 8 qubits) and few iterations first
- **Memory considerations:** For N > 12, consider using MPS methods exclusively (avoid dense conversion)
- **Data storage:** Each experiment creates timestamped files, so multiple runs won't overwrite
- **Analysis:** Use numpy to load saved arrays: `data = np.load('results/.../file.npy')`

## Common Experiment Types

### Thermalization Studies
- Compare integrable (matchgate) vs chaotic (Clifford/Haar) circuits
- Track entanglement entropy growth
- Monitor IPR decay

### Magic/Non-Clifford Studies
- Add T-gates or other non-Clifford gates
- Track FAF growth (should increase for non-Gaussian states)
- Compare to Clifford baseline

### Initial State Dependence
- Vacuum vs product vs random MPS
- Track how initial conditions affect equilibration
- Measure relaxation timescales

### Hybrid Dynamics
- Alternating integrable/chaotic layers
- Studying approach to ergodicity
- Measuring many-body localization transitions
