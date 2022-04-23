import numpy as np
from scipy.stats import pearsonr

def eculidDisSim(x,y):
    '''
    欧几里得相似度
    '''
    return np.sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))

def cosSim(x,y):
    '''
    余弦相似度
    '''
    tmp=np.sum(x*y)
    non=np.linalg.norm(x)*np.linalg.norm(y)
    return np.round(tmp/float(non),9)

def pearsonrSim(x,y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x,y)[0]

def manhattanDisSim(x,y):
    '''
    曼哈顿相似度
    '''
    return sum(abs(a-b) for a,b in zip(x,y))


if __name__=='__main__':
    a=np.array([1,2,3])
    b=np.array([6,5,4])
    sim_eculidDisSim=eculidDisSim(a,b)
    print("sim_eculidDisSim:", sim_eculidDisSim)
    sim_cosSim=cosSim(a,b)
    print("sim_cosSim:", sim_cosSim)
    sim_pearsonrSim=pearsonrSim(a,b)
    print("sim_pearsonrSim:", sim_pearsonrSim)
    sim_manhattanDisSim=manhattanDisSim(a,b)
    print("sim_manhattanDisSim:", sim_manhattanDisSim)

