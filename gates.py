'''
Define all fundamental gates, and also the protocols for generating random 
circuits of local gates, cliffords, matchgates, and more. 
'''
import random
import quimb.tensor as qtn
import numpy as np
from tqdm import tqdm

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
delta_list = [Id, CNOT, CNOTb, SWAP]
sl = [-1, 1]
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])

Pauli_list = [Id, np.kron(X, sId), np.kron(X, Z), np.kron(X,X),
              np.kron(Z, sId), np.kron(Z, Z), np.kron(Z,X),
              np.kron(sId, X), np.kron(sId, Z), np.real(np.kron(Y, Y)),
              np.kron(Y, sId), np.kron(sId, Y), np.kron(Y, Z), np.kron(Z, Y)]

Had_list = [Id, np.kron(H, sId), np.kron(sId, H), np.kron(H,H)]
Swapo = [Id, SWAP]
Phase_l = [np.kron(sId, P), np.kron(P, sId), Id, np.kron(P, P)]

#do I need to fix this?
def Sample_Clifford():

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
    Etta = np.zeros((dim,dim))
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

    pass

def random_fendley(N):

    pass