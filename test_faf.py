"""Test script to debug FAF expectation value computation."""
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from gates import JW, multiply_strings, make_ortho

# Test parameters
N = 4

# Create a simple MPS
state = qtn.MPS_computational_state('0' * N)
for site in range(N-1):
    state.gate_split(make_ortho(4, 0, 1), (site, site+1), inplace=True)

print(f"MPS site_ind_id: {state.site_ind_id}")
print(f"MPS site indices: {[state.site_ind(i) for i in range(N)]}")

# Pauli matrices
pauli_matrices = {
    'I': qu.eye(2),
    'X': qu.pauli('X'),
    'Y': qu.pauli('Y'),
    'Z': qu.pauli('Z')
}

# Test Pauli string
encoding = JW(N)
print(f"\nJW encoding: {encoding}")

# Test one expectation value
i, j = 0, 2
phase, pauli_string = multiply_strings(encoding[i], encoding[j])
pauli_string_padded = pauli_string + 'I' * (N - len(pauli_string))
print(f"\nTesting: encoding[{i}] * encoding[{j}]")
print(f"Result: phase={phase}, string={pauli_string_padded}")

# Build operator list
ops = [pauli_matrices[c] for c in pauli_string_padded]

# Method 1: Try MPO_product_operator with upper/lower indices
print("\n--- Method 1: MPO_product_operator ---")
try:
    pauli_mpo = qtn.MPO_product_operator(ops)
    print(f"MPO created: {pauli_mpo}")
    print(f"MPO site_ind_id: {pauli_mpo.site_ind_id if hasattr(pauli_mpo, 'site_ind_id') else 'N/A'}")

    # Try to match indices
    # The MPO has upper and lower physical indices
    result = state.expec(pauli_mpo)
    print(f"Expectation result type: {type(result)}")
    print(f"Expectation result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Method 2: Using state.expec() with non-identity operators only
print("\n--- Method 2: state.expec() with non-identity ops ---")
try:
    # Find non-identity operators and their sites
    non_id_sites = []
    non_id_ops = []
    for site_idx, char in enumerate(pauli_string_padded):
        if char != 'I':
            non_id_sites.append(site_idx)
            non_id_ops.append(pauli_matrices[char])

    print(f"Non-identity sites: {non_id_sites}")
    print(f"Non-identity ops: {[c for c in pauli_string_padded if c != 'I']}")

    if len(non_id_sites) == 1:
        result = state.expec(non_id_ops[0], non_id_sites[0])
    else:
        gate = non_id_ops[0]
        for op in non_id_ops[1:]:
            gate = np.kron(gate, op)
        result = state.expec(gate, non_id_sites)

    print(f"Expectation result type: {type(result)}")
    print(f"Expectation result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Method 3: Use dense for comparison
print("\n--- Method 3: Dense (for comparison) ---")
try:
    psi_dense = state.to_dense().flatten()

    # Build full operator
    full_op = ops[0]
    for op in ops[1:]:
        full_op = np.kron(full_op, op)

    result = np.vdot(psi_dense, full_op @ psi_dense)
    print(f"Expectation result: {result}")
except Exception as e:
    print(f"Error: {e}")
