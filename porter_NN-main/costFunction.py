import numpy as np
def cost_fnc(pred, y):
    """ pred is a 2d numpy array that consists of probabilities for each observation
        y is a 2d numpy array that consists of labels for each observation"""
        
    #YOUR CODE GOES HERE
    cost_sum = np.sum(np.where(y==0, np.log(1-pred), np.log(pred)))
    cost = (-1 * cost_sum)/len(pred)
    
    return cost.round(2) 
