def ESS(state, N):

    state.canonize(N//2)
    s_d = self.mps.schmidt_values(N//2)
    lt = []
    rt = []
    for g in range(1, len(s_d)-1):
        vals = [s_d[g-1]-s_d[g], s_d[g]-s_d[g+1]]
        lt.append(min(vals)/max(vals))
        rt.append(vals[0]/vals[1])
    lt = [x for x in lt if str(x) != 'nan']
    result += np.average(lt)
    #APPEND RESULT TO NUMPY ARRAY

def repulsion(state):

    state.canonize(N//2)
    s_d = self.mps.schmidt_values(N//2)
    rt = []
    for g in range(1, len(s_d)-1):
        vals = [s_d[g-1]-s_d[g], s_d[g]-s_d[g+1]]
        rt.append(vals[0]/vals[1])
    rt = [x for x in rt if x < 5]
    bins = np.linspace(0,4,201)
    digitized = np.digitize(rt, bins)
    frt = np.array([len(np.array(rt)[digitized==dn]) for dn in range(1, len(bins))]) / len(rt)
    for k in range(200):
        self.rdist[layer, k] = frt[k]

def IPR(state):

    pass #only takes in mps 

def Renyi10(state, N):
    #this is techincally not renyi, but tr(rho^k), need to edit/make new metric
    for k in range(10):
        if k == 1:
            self.renyis[k] += self.mps.entropy(N//2)
        else:
            self.renyis[k] += sum( np.array(self.mps.schmidt_values(N//2)**k) )

def FAF(state):
    
    pass #takes in cov, or can take in state and learn cov 

def bonddim(state):

    pass #takes in mps 

def flatness(state):

    pass #takes in mps

'''
  def get_data(self, layer):
        
        rt = [x for x in rt if x < 4]
        bins = np.linspace(0,4,201)
        digitized = np.digitize(rt, bins)
        frt = np.array([len(np.array(rt)[digitized==dn]) for dn in range(1, len(bins))]) / len(rt)
        for k in range(200):
            self.rdist[layer, k] = frt[k]
'''