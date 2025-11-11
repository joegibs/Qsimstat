def random_product(state):

    if type(state) == mps:
        for site in mps:
            contract_in_place(mps[site], random_gate[site])
    
    if type(state) == cov:
        #import code from previous project on product setups 

    if type(state) == pauliprop:
        pass

    if type(state) == majoranaprop:
        pass

def random_clifford_brickwork(state):

    if type(state) == mps:
        for depth, sites
            contract_random_into_mps #needs random clifford code 
    
    if type(state) == cov:
        #change operators

    if type(state) == pauliprop:
        #change operators

    if type(state) == majoranaprop:
        pass

def (state):

    if type(state) == mps:
        for depth, sites
            contract_random_into_mps #needs random matchgate code
    
    if type(state) == cov:
        #do rotation

    if type(state) == pauliprop:
        pass

    if type(state) == majoranaprop:
        #changes coefficients

def random_matchgate_haar(state):

    if type(state) == mps:
        make_circuit_via_givens 
        apply gates above to mps #needs random mps code
    
    if type(state) == cov:
        #needs previous code

    if type(state) == pauliprop:
        pass #givens rotations into paulis? 

    if type(state) == majoranaprop:
        #changes coefficients

def twolocal(N):

    #do this with DMRG?
    make random 2-local Pauli Hamiltonian 
    compute ground state

def fendleydisguise(N):

    make random claw free frustration graph 
    compute ground state 