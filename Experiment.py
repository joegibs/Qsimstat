'''
#This is all going to be pseudo-code for now. Will fill in later, 
#using previous work and new code. But this will establish file
#structure and workflow. 
'''

from gates import *
import quimb.tensor as qtn
import numpy as np
#import scipy.linalg as lng

class Experiment:

    def __init__(self, N):

        self.N = N
        self.tracktag = {"mps":True, "cov":False, "pauliprop":False, "majprop":False}
        self.statstag = {"renyi10": False, "FAF":False, "IRP":False, "ESS":False, "bonddim":False, "flatness":False}
        self.circuittag = {"depth":0, "iter":0, "sequence":"", "dataC":""}
        self.frames = {}

    def exp_setup(self):

        for tags in [self.tracktag, self.statstag]:
            for k in tags.keys():
                t = input(str(k) + "?: ")
                if t not in ["", " ", "false", "False"]:
                    tags[k] = True

        self.circuittag["depth"] = int(input("depth?: "))
        self.circuittag["iter"] = int(input("iterations?: "))
        self.circuittag["sequence"] = (input("gate sequence? (comma split): ")).split(',')
        self.circuittag["dataC"] = (input("When to take data? (2,10,end): "))

        if self.circuittag["dataC"] == "end":
            num_data_points = 1
        else:
            num_data_points = self.circuittag["depth"] // int(self.circuittag["dataC"])

        for k in self.statstag.keys():
            if self.statstag[k] == True:
                self.frames[k] = np.array((self.circuittag["iter"], num_data_points))


    def execute(self):

        #init data here 

        if "random_product" in self.circuittag["sequence"]:
            self.circuittag["sequence"].pop("random_product")
        depth = self.circuittag["depth"] // (1+len(self.circuittag["sequence"]))

        for it in self.circuittag["iter"]:
            #add state tracking based on circuittag later
            state = qtn.MPS_computational_state('0'*N)
            circuit_list = self.circuittag["sequence"]
            #adjust depth based on instruction list
            for instruction in circuit_list:
                circuit_params = self.run_circuit(instruction, state, depth)
            #calculate final data here

    #for now, only assume mps tracking. later do more. always assume mps tracking for now. 
    def run_circuit(self, instruction, state, depth):

        '''
        more product-like initial states
        elif init_state == 'three_q':
            for i in np.arange(0, self.N, 3):
                for j in range(3):
                    self.mps.gate_split(make_ortho(4,0,1), (i, i+1),inplace=True)
                    self.mps.gate_split(make_ortho(4,0,1), (i+1, i+2),inplace=True)
        elif init_state == 'four_q':
            for i in np.arange(0, self.N, 4):
                for j in range(3):
                    self.mps.gate_split(make_ortho(4,0,1), (i, i+1),inplace=True)
                    self.mps.gate_split(make_ortho(4,0,1), (i+2, i+3),inplace=True)
                    self.mps.gate_split(make_ortho(4,0,1), (i+1, i+2),inplace=True)
'''

        if instruction == "random_product":
            for site in range(self.N):
                state.gate(make_ortho(2,0,1), site, inplace=True,contract=True)

        elif instruction == "random_clifford_brickwork":
            for d in range(depth//2):
                for sites in np.concatenate(np.arange(1,self.N,2),np.arange(2,self.N,2)):
                    state.gate_split(Sample_Clifford(), (sites, sites+1), inplace=True)
                if self.circuittag["dataC"] != 'end':
                    if d*2 % self.circuittag["dataC"] == 0 and d*2 != depth:
                        self.compute_data(state)

        elif instruction == "random_matchgate_brickwork":
            for d in range(depth//2):
                for sites in np.concatenate(np.arange(1,self.N,2),np.arange(2,self.N,2)):
                    state.gate_split(PPgate(), (sites, sites+1), inplace=True)
                if self.circuittag["dataC"] != 'end':
                    if d*2 % self.circuittag["dataC"] == 0 and d*2 != depth:
                        self.compute_data(state)

        elif instruction == "random_matchgate_haar":
            pass #no depth call - givens rotations? 
        elif instruction == "twolocal":
            pass #no data or depth - make random Pauli Hamiltonian, find GS
        elif instruction == "fendleydisguise":
            pass #no data or depth 
            

    def compute_data(self):
    
        pass