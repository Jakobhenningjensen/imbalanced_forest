#Calculates the gini impurity
import numpy as np

def gini(node_targets):
    #Different targets
    _,counts = np.unique(node_targets,return_counts=True)
    #Number of observations
    n_obs = len(node_targets)
    frac = np.array([c/n_obs for c in counts])
    gini = 1-np.sum(frac**2)

    return gini


def entropy(node_targets):
    _,counts = np.unique(node_targets,return_counts=True)
    #Number of observations
    n_obs = len(node_targets)
    frac = [np.log2(c/n_obs)*(c/n_obs) for c in counts]
    entropy = -sum(frac)

    return entropy
