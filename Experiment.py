'''
#This is all going to be pseudo-code for now. Will fill in later, 
#using previous work and new code. But this will establish file
#structure and workflow. 
'''

from gates import *
from DataFunctions import *
import quimb.tensor as qtn
import numpy as np
#import scipy.linalg as lng

class Experiment:

    def __init__(self, N):
        """Initialize an Experiment instance.

        Args:
            N (int): Number of qubits in the system.

        Attributes:
            N (int): Number of qubits.
            tracktag (dict): Dictionary tracking which state representations to maintain.
            statstag (dict): Dictionary tracking which statistics to compute.
            circuittag (dict): Dictionary storing circuit configuration parameters.
            frames (dict): Dictionary storing data arrays for each statistic.
        """

        self.N = N
        self.tracktag = {"mps":True, "cov":False, "pauliprop":False, "majprop":False}
        self.statstag = {"renyi10": False, "faf":False, "ipr":False, "ess":False, "bonddim":False,
                         "flatness":False, "repulsion":False, "sff":False, "porter_thomas":False, "sre":False}
        self.circuittag = {"depth":0, "iter":0, "sequence":"", "dataC":""}
        self.frames = {}
        self.encoding = JW(self.N) # Will be set during exp_setup if FAF is enabled

    def exp_setup(self):
        """Set up experiment configuration through interactive user input.

        Prompts the user to configure:
        - Which state representations to track (tracktag)
        - Which statistics to compute (statstag)
        - Circuit parameters (depth, iterations, gate sequence, data collection frequency)

        Initializes data frames (numpy arrays) for each enabled statistic based on
        the number of iterations and data collection points.

        Returns:
            None
        """

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
                if k == "renyi10":
                    self.frames[k] = np.zeros((self.circuittag["iter"], num_data_points, 10))
                elif k == "faf":
                    self.frames[k] = np.zeros((self.circuittag["iter"], num_data_points, 10))
                elif k == "repulsion":
                    self.frames[k] = np.zeros((self.circuittag["iter"], num_data_points, 200))
                elif k == "bonddim":
                    self.frames[k] = np.zeros((self.circuittag["iter"], num_data_points, self.N-1))
                elif k == "porter_thomas":
                    # Porter-Thomas returns a dict with multiple arrays, store as object array
                    self.frames[k] = np.empty((self.circuittag["iter"], num_data_points), dtype=object)
                elif k in ["ipr", "ess", "flatness", "sff", "sre"]:
                    self.frames[k] = np.zeros((self.circuittag["iter"], num_data_points))
                else:
                    self.frames[k] = np.zeros((self.circuittag["iter"], num_data_points))



    def execute(self):
        """Execute the quantum circuit experiment.

        Runs the circuit evolution for the specified number of iterations,
        applying gate sequences according to the circuit configuration.
        Computes and stores statistics at specified intervals.

        Returns:
            None

        Note:
            Currently assumes MPS state tracking only.
        """

        if "random_product" in self.circuittag["sequence"]:
            self.circuittag["sequence"].pop("random_product")
        depth = self.circuittag["depth"] // (1+len(self.circuittag["sequence"]))

        for it in self.circuittag["iter"]:
            state = qtn.MPS_computational_state('0'* self.N)
            circuit_list = self.circuittag["sequence"]
            #adjust depth based on instruction list
            for instruction in circuit_list:
                circuit_params = self.run_circuit(instruction, state, depth)
            #calculate final data here

        # Save all data after iterations complete
        print("\nSaving data to disk...")
        self.save_data()

    #for now, only assume mps tracking. later do more. always assume mps tracking for now.
    def run_circuit(self, instruction, state, depth, iter):
        """Run a specific circuit instruction on the quantum state.

        Args:
            instruction (str): Type of circuit to run. Options include:
                - "random_product": Random product state preparation
                - "random-mps": Random mps state prep
                - "random_clifford_brickwork": Brickwork circuit with random Clifford gates
                - "random_matchgate_brickwork": Brickwork circuit with random matchgates
                - "random_haar_brickwork": Brickwork circuit with random Haar gates
                - "random_matchgate_haar": Random matchgate Haar circuit
                - "twolocal": Two-local Hamiltonian ground state preparation
                - "fendleydisguise": Fendley disguise protocol
            state (qtn.MPS): Current quantum state as Matrix Product State.
            depth (int): Circuit depth for this instruction.
            iter (int): Current iteration number for data storage.

        Returns:
            None: Modifies state in-place and stores computed statistics.

        Note:
            Currently assumes MPS tracking only. Data is collected at intervals
            specified by self.circuittag["dataC"].

            More product-like initial states can be added (e.g., three_q, four_q).
        """

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

        if instruction == "random_mps":
            state = qtn.gen.rand.rand_matrix_product_state(self.N, self.N**2 )

        #will this work with sites? how do sites work here?
        if instruction == "random_mera":
            state = qtn.gen.rand.rand_mera(self.N, self.N**2 )

        elif instruction == "random_clifford_brickwork":
            for d in range(depth//2):
                for sites in np.concatenate([np.arange(0,self.N-1,2),np.arange(1,self.N-1,2)]):
                    state.gate_split(Sample_Clifford(), (sites, sites+1), inplace=True)
                if self.circuittag["dataC"] != 'end':
                    if d*2 % self.circuittag["dataC"] == 0 and d*2 != depth:
                        self.compute_data(state, iter, d*2 // self.circuittag["dataC"])

        elif instruction == "random_matchgate_brickwork":
            for d in range(depth//2):
                for sites in np.concatenate([np.arange(0,self.N-1,2),np.arange(1,self.N-1,2)]):
                    state.gate_split(PPgate(), (sites, sites+1), inplace=True)
                if self.circuittag["dataC"] != 'end':
                    if d*2 % self.circuittag["dataC"] == 0 and d*2 != depth:
                        self.compute_data(state, iter, d*2 // self.circuittag["dataC"])

        elif instruction == "random_haar_brickwork":
            for d in range(depth//2):
                for sites in np.concatenate([np.arange(0,self.N-1,2),np.arange(1,self.N-1,2)]):
                    state.gate_split(make_ortho(4,0,1), (sites, sites+1), inplace=True)
                if self.circuittag["dataC"] != 'end':
                    if d*2 % self.circuittag["dataC"] == 0 and d*2 != depth:
                        self.compute_data(state, iter, d*2 // self.circuittag["dataC"])

        elif instruction == "random_matchgate_haar":
            pass #no depth call - givens rotations? 
        elif instruction == "twolocal":
            pass #no data or depth - make random Pauli Hamiltonian, find GS
        elif instruction == "fendleydisguise":
            pass #no data or depth 
            

    def compute_data(self, state, it, point):
        """Compute and store statistics for the current quantum state.

        Args:
            state (qtn.MPS): Current quantum state as Matrix Product State.
            it (int): Current iteration number for data indexing.
            point (int): Data collection point index within this iteration.

        Returns:
            None: Stores computed statistics in self.frames arrays.

        Note:
            Only computes statistics that are enabled in self.statstag.
            Available statistics: renyi10, faf, ipr, ess, bonddim, flatness, repulsion,
            sff, porter_thomas, sre.
        """

        if self.statstag["renyi10"] == True:
            self.frames["renyi10"][it, point] = Renyi10(state, self.N)
        if self.statstag["faf"] == True:
            faf_values = np.zeros(10)
            for k in range(1, 11):
                faf_values[k-1] = FAF(state, self.encoding, self.N, k)
            self.frames["faf"][it, point] = faf_values
        if self.statstag["ipr"] == True:
            self.frames["ipr"][it, point] = IPR(state)
        if self.statstag["ess"] == True:
            self.frames["ess"][it, point] = ESS(state, self.N)
        if self.statstag["bonddim"] == True:
            self.frames["bonddim"][it, point] = bonddim(state)
        if self.statstag["flatness"] == True:
            self.frames["flatness"][it, point] = flatness(state, self.N)
        if self.statstag["repulsion"] == True:
            self.frames["repulsion"][it, point] = repulsion(state, self.N)
        if self.statstag["sff"] == True:
            self.frames["sff"][it, point] = SFF(state, self.N)
        if self.statstag["porter_thomas"] == True:
            self.frames["porter_thomas"][it, point] = porter_thomas(state, self.N)
        if self.statstag["sre"] == True:
            self.frames["sre"][it, point] = SRE(state, self.N)

    def save_data(self, base_dir="results"):
        """Save all computed data frames to numpy arrays.

        Creates a directory structure: results/metric_name/N_iter_dataC_date.npy
        for each enabled statistic.

        Args:
            base_dir (str): Base directory for saving results. Default is "results".

        Returns:
            None: Saves arrays to disk.

        Note:
            Creates subdirectories for each metric if they don't exist.
            File naming: {N}qubits_{iter}iter_{dataC}interval_{timestamp}.npy
        """
        import os
        from datetime import datetime

        # Create base results directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename components
        N_str = f"{self.N}qubits"
        iter_str = f"{self.circuittag['iter']}iter"
        dataC_str = f"{self.circuittag['dataC']}interval"
        filename_base = f"{N_str}_{iter_str}_{dataC_str}_{timestamp}"

        # Save each enabled metric
        for metric_name, is_enabled in self.statstag.items():
            if is_enabled and metric_name in self.frames:
                # Create metric subdirectory
                metric_dir = os.path.join(base_dir, metric_name)
                if not os.path.exists(metric_dir):
                    os.makedirs(metric_dir)

                # Save array
                filepath = os.path.join(metric_dir, f"{filename_base}.npy")
                np.save(filepath, self.frames[metric_name])
                print(f"  Saved {metric_name}: {filepath}")

        print(f"\nAll data saved to {base_dir}/")