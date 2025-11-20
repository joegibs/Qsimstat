'''
Define all fundamental gates, and also the protocols for generating random 
circuits of local gates, cliffords, matchgates, and more. 
'''
import random
import quimb.tensor as qtn
import numpy as np
from tqdm import tqdm
import random

#fundamental gates for simulation
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
P = np.array([[1,0],[0,1j]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CNOTb = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
Id = np.identity(4)
sId = np.identity(2)
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])

#list of gates needed for random clifford generation
delta_list = [Id, CNOT, CNOTb, SWAP]
sl = [-1, 1]

Pauli_list = [Id, np.kron(sId, X), np.kron(sId, Z),np.kron(sId, Y),
            np.kron(X, sId), np.kron(X, X), np.kron(X, Z), np.kron(X,Y),
            np.kron(Z, sId), np.kron(Z, Z), np.kron(Z,X),np.kron(Z, Y),
            np.kron(Y, sId),np.real(np.kron(Y, Y)),np.kron(Y, Z), np.kron(Y, X)]
Had_list = [Id, np.kron(H, sId), np.kron(sId, H), np.kron(H,H)]
Swapo = [Id, SWAP]
Phase_l = [np.kron(sId, P), np.kron(P, sId), Id, np.kron(P, P)]

#do I need to fix this?
def Sample_Clifford():
    """Generate a random 2-qubit Clifford gate.

    Samples a random Clifford gate by composing Pauli operators, phase gates,
    Hadamard gates, and delta operations (CNOT, SWAP, etc.).

    Returns:
        np.ndarray: 4x4 unitary matrix representing a random Clifford gate.

    Note:
        Uses the decomposition: Cliff = sign * F2 @ h @ F1, where
        F1, F2 are compositions of delta, Pauli, and phase operations,
        h is a composition of Hadamard and SWAP gates, and sign is ±1.
    """

    F1= np.matmul(random.choice(delta_list), random.choice(Pauli_list))
    F1 = np.matmul(F1, random.choice(Phase_l))
    F2= np.matmul(random.choice(Pauli_list), random.choice(delta_list))
    F2 = np.matmul(F2, random.choice(Phase_l))
    h = np.matmul(random.choice(Had_list), random.choice(Swapo))
    Cliff = random.choice(sl)*np.matmul(F2, np.matmul(h, F1))
    return Cliff


def make_sim(dim, simmean, simwidth):
    """
    Makes randomly matrix in GL(N, C). This matrix can be assumed to be 
    invertible because the measure of non-invertible matrices when 
    randomly selecting from C(N) is zero

    Parameters
    ----------
    dim : int
        dimension of matrix
    simmean : int
        mean of distribution random complex variables are chosen from
    simwidth : int
        width of distribution random complex variables are chosen from

    Returns
    -------
    SM : array
        matrix in GL(N, C)

    """
    
    RR = np.random.normal(simmean, simwidth, (dim,dim))
    SM = RR + 1j * np.random.normal(simmean, simwidth, (dim,dim))
    return SM

def make_unitary(dim, simmean, simwidth):
    """
    Generates unitary matrix via QR decomposition of matrix in GL(N, C)
    See parameters above

    Returns
    -------
    U : array
        unitary array

    """
    sim = make_sim(dim, simmean, simwidth)
    Q, R = np.linalg.qr(sim)
    Etta = np.zeros((dim,dim), dtype=complex)
    for j in range(dim):
        Etta[j,j] = R[j,j]/np.linalg.norm(R[j,j])
    U = np.matmul(Q, Etta)
    return U

def make_ortho(dim, simmean, simwidth):
    """
    Generates unitary matrix via QR decomposition of matrix in GL(N, C)
    See parameters above

    Returns
    -------
    U : array
        unitary array

    """
    sim = np.random.normal(simmean, simwidth, (dim,dim))
    Q, R = np.linalg.qr(sim)
    return -1*Q

def dephase(unitary):
    """
    Dephases unitary, turns it from U(N) to SU(N)

    Parameters
    ----------
    unitary : array
        input matrix in U(N)

    Returns
    -------
    unitary : array
        depahsed matrix in SU(N)

    """
    
    glob = np.linalg.det(unitary)
    theta = np.arctan(np.imag(glob) / np.real(glob)) / 2
    unitary = unitary * np.exp(-1j*theta)
    if np.round(np.linalg.det(unitary)) < 0:
        unitary = unitary * 1j
    return unitary

def PPgate():
    """Generate a random matchgate (particle-preserving gate).

    Creates a 2-qubit matchgate that preserves particle number, constructed
    from two random SU(2) unitaries in the particle-preserving subspace.

    Returns:
        np.ndarray: 4x4 matchgate matrix with particle-preserving structure.
            Has the form with zeros in positions that violate particle conservation.

    Note:
        Matchgates are equivalent to free fermion evolution and preserve
        the computational basis weight. Both u1 and u2 are dephased to SU(2).
    """

    u1 = make_unitary(2, 0, 1)
    u2 = make_unitary(2, 0, 1)
    u1 = dephase(u1)
    u2 = dephase(u2)
    G_AB = np.array([[u1[0,0], 0, 0, u1[0,1]],
                      [0, u2[0,0], u2[0,1], 0],
                      [0, u2[1,0], u2[1,1], 0],
                      [u1[1, 0], 0, 0, u1[1,1]]])
    return G_AB

def random_twolocal(N):
    """Generate a random two-local Hamiltonian.

    Creates a Hamiltonian with random two-qubit interactions between
    neighboring qubits.

    Args:
        N (int): Number of qubits in the system.

    Returns:
        np.ndarray or qtn.Tensor: Two-local Hamiltonian (to be implemented).

    Note:
        Currently not implemented. Will generate random Pauli Hamiltonian
        and potentially find its ground state.
    """

    pass

def random_fendley(N):
    """Generate a random circuit in Fendley's disguise protocol.

    Implements Fendley's protocol for disguising free fermion circuits
    to appear non-integrable while maintaining integrability.

    Args:
        N (int): Number of qubits in the system.

    Returns:
        circuit or list: Fendley disguise circuit (to be implemented).

    Note:
        Currently not implemented. Will implement the Fendley disguise
        technique for creating integrable circuits that mimic chaotic behavior.
    """

    pass


def JW(N):

    """Generate the Jordan-Wigner majoranas as text strings for given N.

    Args:
        N (int): Number of qubits in the system.

    Returns:
        list: list of Jordan-Wigner strings 

    Note:
        In future, want to include group code on inverse mappings between encodings
        And does quimb need I's for empty indices on edges? 
    """
    JWl = []
    for i in range(N):
        JWl.append('Z'*i + 'X')
        JWl.append('Z'*i + 'Y')
    return JWl

    #quimb.Pauli(string) should work 

def multiply_paulis(p1, p2):
    """Multiply two single-qubit Pauli operators.

    Args:
        p1 (str): First Pauli operator ('I', 'X', 'Y', or 'Z')
        p2 (str): Second Pauli operator ('I', 'X', 'Y', or 'Z')

    Returns:
        tuple: (phase, result) where phase is 1, -1, 1j, or -1j and result is the Pauli string
    """
    # Identity cases
    if p1 == 'I':
        return 1, p2
    if p2 == 'I':
        return 1, p1

    # Same Pauli -> Identity
    if p1 == p2:
        return 1, 'I'

    # Different Paulis - need to track phase
    # X*Y = iZ, Y*X = -iZ
    # Y*Z = iX, Z*Y = -iX
    # Z*X = iY, X*Z = -iY

    pauli_mult = {
        ('X', 'Y'): (1j, 'Z'),
        ('Y', 'X'): (-1j, 'Z'),
        ('Y', 'Z'): (1j, 'X'),
        ('Z', 'Y'): (-1j, 'X'),
        ('Z', 'X'): (1j, 'Y'),
        ('X', 'Z'): (-1j, 'Y'),
    }

    return pauli_mult[(p1, p2)]


def multiply_strings(s1, s2):
    """Multiply two Pauli strings and track the phase.

    Args:
        s1 (str): First Pauli string (e.g., 'XYZ', 'ZX', 'Y')
        s2 (str): Second Pauli string (same length as s1)

    Returns:
        tuple: (phase, result_string) where phase is complex (1, -1, 1j, -1j)
               and result_string is the resulting Pauli string

    Note:
        Strings are padded with 'I' if they have different lengths.
    """
    # Pad shorter string with 'I' to match lengths
    max_len = max(len(s1), len(s2))
    s1 = s1 + 'I' * (max_len - len(s1))
    s2 = s2 + 'I' * (max_len - len(s2))

    total_phase = 1
    result = ''

    for p1, p2 in zip(s1, s2):
        phase, pauli = multiply_paulis(p1, p2)
        total_phase *= phase
        result += pauli

    return total_phase, result


def random_pauli(N):

    """Generates a random Pauli string

    Args:
        N (int): Number of qubits in the system.

    Returns:
        string: random Pauli string
    """

    rand_pauli = ""
    for _ in range(N):
        rand_pauli += random.choice(["X", "Y", "Z", "I"])
    return rand_pauli