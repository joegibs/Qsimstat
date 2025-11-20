import numpy as np
import quimb as qu

def pauli_expectation(state, pauli_string):
    """Compute expectation value of a Pauli string using MPS.

    Computes <psi|P|psi> by applying gates and computing overlap.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        pauli_string (str): Pauli string like "XZIY" (length must match number of qubits).

    Returns:
        complex: Expectation value <psi|P|psi>.

    Note:
        This uses the gate-and-overlap method. A compatibility issue with quimb 1.11.2
        prevents using state.local_expectation() directly. Upgrading quimb or fixing
        the partial_trace API issue would allow further optimization.
    """
    # Pauli matrices
    pauli = {
        'I': qu.pauli('I'),
        'X': qu.pauli('X'),
        'Y': qu.pauli('Y'),
        'Z': qu.pauli('Z')
    }

    # Find non-identity sites and their operators
    ops = []
    where = []
    for i, ch in enumerate(pauli_string):
        if ch == 'I':
            continue
        ops.append(pauli[ch])
        where.append(i)

    # All identities -> expectation is 1
    if not ops:
        return 1.0

    # Compute <psi|P|psi> by applying gates to a copy and computing overlap
    psi_copy = state.copy()
    for op, site in zip(ops, where):
        psi_copy.gate_(op, site)

    # Compute overlap <psi|P|psi>
    return state.H @ psi_copy

def ESS(state, N):
    """Calculate the Entanglement Spectrum Statistics (ESS).

    Computes the ratio of consecutive gaps in the Schmidt spectrum to
    characterize the entanglement structure.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        N (int): Number of qubits in the system.

    Returns:
        float: Average ESS value computed from Schmidt value gaps.

    Note:
        Canonizes the state at the center (N//2) before computing Schmidt values.
        Filters out NaN values from the calculation.
    """

    state.canonize(N//2)
    s_d = state.schmidt_values(N//2)
    lt = []
    rt = []
    for g in range(1, len(s_d)-1):
        vals = [s_d[g-1]-s_d[g], s_d[g]-s_d[g+1]]
        lt.append(min(vals)/max(vals))
        rt.append(vals[0]/vals[1])
    lt = [x for x in lt if str(x) != 'nan']
    result = np.average(lt)
    return result

def repulsion(state, N):
    """Calculate the level repulsion distribution of Schmidt gaps.

    Computes the distribution of ratios between consecutive Schmidt value gaps,
    which characterizes quantum chaos and thermalization behavior.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        N (int): Number of qubits in the system.

    Returns:
        np.ndarray: Normalized frequency distribution of gap ratios binned
            into 200 bins between 0 and 4.

    Note:
        Canonizes the state at the center (N//2) before computing Schmidt values.
        Filters out outlier ratios greater than 5.
    """

    state.canonize(N//2)
    s_d = state.schmidt_values(N//2)
    rt = []
    for g in range(1, len(s_d)-1):
        vals = [s_d[g-1]-s_d[g], s_d[g]-s_d[g+1]]
        rt.append(vals[0]/vals[1])
    rt = [x for x in rt if x < 5]
    bins = np.linspace(0,4,201)
    digitized = np.digitize(rt, bins)
    if len(rt) == 0:
        return np.zeros(200)
    frt = np.array([len(np.array(rt)[digitized==dn]) for dn in range(1, len(bins))]) / len(rt)
    return frt

def IPR(state):
    """Calculate the Inverse Participation Ratio (IPR).

    Measures the localization of the quantum state in the computational basis.
    A smaller IPR indicates more localization.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.

    Returns:
        float: IPR value computed as sum of |c_i|^4 where c_i are state coefficients.

    Note:
        Converts MPS to dense state vector to compute IPR.
        IPR = 1 for completely localized state, 1/2^N for maximally delocalized.
    """

    # Convert MPS to dense state vector
    psi = state.to_dense()
    # Compute IPR as sum of |c_i|^4
    ipr = np.sum(np.abs(psi)**4)
    return ipr 

def Renyi10(state, N):
    """Calculate Renyi entropies (technically tr(rho^k)) for k=0 to 9.

    Computes the trace of powers of the reduced density matrix for various orders,
    which characterizes the entanglement structure.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        N (int): Number of qubits in the system.

    Returns:
        np.ndarray: Array of 10 values, where index k contains tr(rho^k).
            For k=1, returns the von Neumann entropy.

    Note:
        This is technically tr(rho^k), not the standard Renyi entropy definition.
        Consider renaming or creating a separate metric for true Renyi entropies.
    """
    #this is techincally not renyi, but tr(rho^k), need to edit/make new metric
    renyis = np.zeros(10)
    for k in range(10):
        if k == 1:
            renyis[k] += state.entropy(N//2)
        else:
            renyis[k] += sum( np.array(state.schmidt_values(N//2)**k) )
    return renyis

def FAF(state, encoding, N, k):
    """Calculate the Fermionic Anti-Flatness (FAF).

    Measures non-Gaussianity of the state in a fermionic encoding by computing
    deviations from Gaussian structure via the covariance matrix.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        encoding (list): Set of 2N majoranas/Pauli operators for the encoding (as strings).
        N (int): Number of qubits in the system.
        k (int): Power to which the covariance matrix is raised.

    Returns:
        float: FAF value computed as N - 1/2 * tr((cov.T @ cov)^k).
            Measures deviation from Gaussian (free fermion) behavior.

    Note:
        Currently only takes MPS input. Future versions may accept covariance
        matrix directly or compute it from the state.
        Assumes symbolic encoding of 2N majoranas/Paulis.
    """
    from gates import multiply_strings

    # Build covariance matrix using MPS expectation values
    cov = np.zeros((2*N, 2*N), dtype=complex)

    for i in range(2*N):
        for j in range(i+1, 2*N):
            # Multiply Pauli strings and get phase
            phase, pauli_string = multiply_strings(encoding[i], encoding[j])
            # Pad the Pauli string to N qubits with identity operators
            pauli_string_padded = pauli_string + 'I' * (N - len(pauli_string))

            # Compute expectation value using MPS local_expectation
            expectation = pauli_expectation(state, pauli_string_padded)
            # Take real part to avoid numerical imaginary artifacts
            expectation = np.real(expectation)
            cov[i,j] = phase * expectation
            # Make antisymmetric: cov[j,i] = -cov[i,j]
            cov[j,i] = -phase * expectation

    # For free fermions (Gaussian states), the FAF formula gives:
    # FAF_k = N - 1/2 * Tr[(Γ^T Γ)^k] where Γ is the antisymmetric covariance matrix
    # For a Gaussian state, this should equal 0
    result = N - 0.5 * np.trace( np.linalg.matrix_power(cov.T @ cov, k) )
    return np.real(result)

def bonddim(state):
    """Calculate the bond dimensions of the MPS.

    Returns the bond dimensions (virtual dimensions) at each bond of the
    Matrix Product State, which characterizes entanglement growth.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.

    Returns:
        np.ndarray: Array of bond dimensions at each position in the MPS.

    Note:
        Bond dimensions indicate the amount of entanglement across each cut.
        Larger bond dimensions require more computational resources.
    """

    # Get bond dimensions from the MPS structure
    # For each tensor, the bond dimension is typically the first or middle dimension
    return np.array(state.bond_sizes()) 

def flatness(state, N):
    """Calculate the flatness of the entanglement spectrum.

    Measures how uniform the Schmidt values are across the entanglement cut,
    which indicates the degree of entanglement spreading.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        N (int): Number of qubits in the system.

    Returns:
        float: Flatness metric computed as exp(-sum(p_i * log(p_i))) / len(p_i),
            where p_i are the normalized squared Schmidt values.

    Note:
        Flatness = 1 indicates perfectly flat (maximally entangled) spectrum.
        Flatness close to 0 indicates peaked (less entangled) spectrum.
    """

    state.canonize(N//2)
    schmidt_vals = state.schmidt_values(N//2)
    # Normalize Schmidt values (they should already be normalized, but ensure it)
    probs = schmidt_vals**2
    probs = np.trim_zeros(probs / np.sum(probs))
    # Calculate entanglement entropy
    entropy = -np.sum(probs * np.log(probs))
    # Flatness is the ratio of actual entropy to maximum possible entropy
    max_entropy = np.log(len(probs))
    flatness_val = np.exp(entropy) / len(probs) if len(probs) > 0 else 0
    return flatness_val

def porter_thomas(state, N):
    """Analyze the Porter-Thomas distribution of state coefficients.

    Compares the distribution of state coefficient magnitudes to the expected
    Porter-Thomas distribution, which is a signature of quantum chaos.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        N (int): Number of qubits in the system.

    Returns:
        dict: Dictionary containing:
            'dkl': Kullback-Leibler divergence from Porter-Thomas distribution
            'coeffs': Sorted absolute values of state coefficients
            'histogram': Binned histogram of coefficient distribution

    Note:
        Porter-Thomas distribution: P(x) = N * exp(-N*x) where x = |c_i|^2 * 2^N
        This is the expected distribution for chaotic quantum states.
    """

    # Convert MPS to dense state vector
    psi = state.to_dense()

    # Compute magnitude squared of each coefficient
    probs = np.abs(psi)**2

    # Sort coefficients
    sorted_probs = np.sort(probs)[::-1]  # descending order

    # Rescale to Porter-Thomas variable: x = |c_i|^2 * 2^N
    dim = 2**N
    x_vals = sorted_probs * dim

    # Create histogram
    bins = np.linspace(0, np.max(x_vals), 100)
    hist, bin_edges = np.histogram(x_vals, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Expected Porter-Thomas distribution: P(x) = exp(-x)
    porter_thomas_dist = np.exp(-bin_centers)
    porter_thomas_dist = porter_thomas_dist / np.sum(porter_thomas_dist)  # normalize
    hist_normalized = hist / np.sum(hist)

    # Calculate Kullback-Leibler divergence
    # DKL = sum(P(x) * log(P(x) / Q(x)))
    # Add small epsilon to avoid log(0)
    eps = 1e-16
    dkl = np.sum(hist_normalized * np.log((hist_normalized + eps) / (porter_thomas_dist + eps)))

    return {
        'dkl': dkl,
        'coeffs': sorted_probs,
        'histogram': hist_normalized,
        'bin_centers': bin_centers
    }

def SRE(state, N, alpha=2, num_samples=1000):
    """Calculate the Stabilizer Renyi Entropy (SRE).

    Computes the stabilizer Renyi entropy which measures magic (non-stabilizerness)
    by computing expectation values of Pauli strings and their Renyi entropy.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        N (int): Number of qubits in the system.
        alpha (float): Renyi parameter. Default is 2.
        num_samples (int): Number of random Pauli strings to sample. Default is 1000.

    Returns:
        float: Stabilizer Renyi entropy M_alpha = (1/(1-alpha)) * log(sum_P |<P>|^(2*alpha))
            where the sum is over Pauli strings P and <P> is the expectation value.

    Note:
        SRE quantifies the "magic" or non-Clifford resources in a quantum state.
        Stabilizer states (Clifford states) have SRE = 0.
        Highly non-stabilizer states have larger SRE values.
        Implementation uses random sampling of Pauli strings for efficiency.
    """

    from gates import random_pauli

    # Sample random Pauli strings and compute their expectation values
    pauli_expectations = []

    for _ in range(num_samples):
        # Generate a random Pauli string
        pauli_string = random_pauli(N)

        # Compute expectation value <psi|P|psi> using MPS local_expectation
        expectation = pauli_expectation(state, pauli_string)
        pauli_expectations.append(expectation)

    # Convert to numpy array and compute |<P>|^2
    pauli_probs = np.abs(np.array(pauli_expectations))**2

    # Normalize the distribution
    pauli_probs = np.array(pauli_probs / np.sum(pauli_probs))

    # Compute Stabilizer Renyi Entropy
    # M_alpha = (1/(1-alpha)) * log(sum_P |<P>|^(2*alpha))
    if alpha == 1:
        # von Neumann-like entropy (limit case)
        # M_1 = -sum(p * log(p))
        pauli_probs = np.trim_zeros(pauli_probs)
        sre = -np.sum(pauli_probs * np.log(pauli_probs))
    else:
        # Renyi entropy
        sre = (1.0 / (1.0 - alpha)) * np.log(np.sum(pauli_probs**alpha))

    return sre

def SFF(state, N):
    """Calculate the Spectral Form Factor (SFF) from the entanglement spectrum.

    Computes the spectral form factor which characterizes the correlations in
    the energy level spacing and is a key diagnostic of quantum chaos.

    Args:
        state (qtn.MPS): Quantum state as Matrix Product State.
        N (int): Number of qubits in the system.

    Returns:
        float: Spectral form factor computed as |sum(exp(i*theta_j))|^2 / n^2
            where theta_j are derived from the entanglement spectrum gaps.

    Note:
        In quantum chaos, SFF shows characteristic "ramp" and "plateau" behavior.
        This implementation uses entanglement spectrum gaps as proxy for energy levels.
        For full SFF analysis, typically need time evolution and averaging.
    """

    state.canonize(N//2)
    schmidt_vals = state.schmidt_values(N//2)

    # Compute gaps in the entanglement spectrum (analogous to energy level spacings)
    gaps = np.diff(schmidt_vals)

    # Map gaps to phases (cumulative sum gives phase-like quantity)
    phases = np.cumsum(gaps)

    # Calculate spectral form factor
    # SFF(t) = |sum_j exp(i * E_j * t)|^2 / N^2
    # Here we use a single "time" slice with phases from gaps
    n = len(phases)
    if n == 0:
        return 0.0

    sff_sum = np.sum(np.exp(1j * phases))
    sff = np.abs(sff_sum)**2 / (n**2)

    return sff