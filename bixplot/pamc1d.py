#basics
import numpy as np
import pandas as pd
import os
from collections import Counter
import math
import warnings

#stats
from sklearn_extra.cluster import KMedoids #kmedoids
from sklearn.exceptions import ConvergenceWarning
import pulp # linear programming package

def pamc1d(y, k, solver=None, minsize=4, countwhat="unique", stand=False,
           maxit=100, verbose=True, random_state=None):
    """
    Version of PAM for univariate data, with the constraint 
    that each cluster contains at least "minsize" cases. 
    - with countwhat = "any" we count any cases, whether they
      are tied or not. This may put tied cases in different clusters.
    - with countwhat = "unique" we only count the unique cases.
      This will never put tied cases in different clusters, 
      but as a result some clusters may contain more cases than with
      option "any".

    This is a modification of an algorithm for constrained k-means 
    by Bradley–Bennett–Demiriz (2000, Microsoft Research Report MSR-TR-2000-65).  
    The modifications include:
      – working with medians (k-medoids) instead of means (k-means)  
      – providing a minimum cluster size constraint (`minsize`)  
      – applying the minsize constraint either to unique values 
        (`countwhat="unique"`) or to all values (`countwhat="any"`)  
      – preventing tied values (duplicates) from being split across clusters 
        when `countwhat="unique"`

    It preserves the original order and length of y (NaN positions will get NaN as 
    clustering id)  

    Parameters
    ----------
    y : array-like
        Univariate data vector (can include NaN).
    k : int
        Number of clusters to form.
    minsize : int, default=4
        Lower bound on the size of all clusters.
    countwhat : "unique", "any", default="unique"
        Whether to enforce `minsize` on unique values or on all values.
    stand : bool, default=True
        If True, standardize (mean=0, sd=1) before clustering.
    maxit : int, default=100
        Maximum number of iterations in the constrained clustering loop.
    verbose : bool, default=True
        If True, prints intermediate steps and objectives.
    random_state : int or None
        Random seed for reproducibility of the initial PAM solution.

    Returns
    -------
    dict
        A dictionary containing:
        – iter : number of iterations performed
        – converged : whether the algorithm converged
        – clustering : cluster labels aligned with original y 
                       (NaN where y was NaN)
        – centers : final cluster medoids
        – sizes : dictionary of cluster sizes
        – objective : final mean L1 distance to medoids
        – objectives : list of objective values across iterations
    """

    def objL1(yy, centers, clusvec):
        """
        Compute mean L1 (absolute) distance of points (in yy) to their assigned cluster medoids (centers). 
        clusvec is a list of the integer cluster assignment for each point in yy.
        """
        return np.mean([abs(yy[i] - centers[c]) for i, c in enumerate(clusvec)])

    #---------------------------------------------------------------------------------------------        
    # --- Preprocessing and bookkeeping
    #---------------------------------------------------------------------------------------------
    y = np.asarray(y, dtype=float)
    n_total = len(y)
    nan_mask = np.isnan(y)

    # indices of non-NA values
    valid_idx = np.where(~nan_mask)[0]
    y_valid = y[valid_idx]

    # store sorting order of the valid part
    sort_idx = np.argsort(y_valid)
    y_sorted = y_valid[sort_idx]
    n = len(y_sorted)
    nuniq = len(set(y_sorted))
        
    if len(np.unique(y_sorted)) < 2:
        raise ValueError("y has < 2 non-NA unique values, so no clustering")

    # standardize if requested
    if stand and np.std(y_sorted) > 0:
        y_sorted = (y_sorted - np.mean(y_sorted)) / np.std(y_sorted, ddof=1)

    if countwhat == "unique" and nuniq == n:
        countwhat = "any"
    if countwhat not in ("any", "unique"):
        warnings.warn("countwhat must be 'any' or 'unique'", stacklevel=2)
        raise ValueError("countwhat must be 'any' or 'unique'")
        
    # feasibility checks
    if countwhat == "any" and k * minsize > n:
        raise ValueError(f"Need {k*minsize} points but only {n} available")    
    
    elif countwhat == "unique" and k * minsize > nuniq:
        raise ValueError(f"Need {k*minsize} unique points but only {nuniq} available")    
    
    elif countwhat not in ("any", "unique"):   
        raise ValueError("countwhat must be 'any' or 'unique'")

        
    #---------------------------------------------------------------------------------------------
    # --- Step 1: Initial clustering with PAM
    #---------------------------------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pam = KMedoids(n_clusters=k, random_state=random_state, method="pam", init="build")
        pam.fit(y_sorted.reshape(-1, 1))
    clusvec = pam.labels_
    centers = pam.cluster_centers_.flatten()

    objectives = []
    obj = objL1(y_sorted, centers, clusvec)
    objectives.append(obj)
    
    #test if conditions are satisfied or if we need the constrain pam
    bigenough = True
    li_clusters = [np.where(clusvec == h)[0] for h in range(k)] #only go through all once
    for h in range(k):
        yh = y_sorted[li_clusters[h]]
        if (countwhat == "any" and len(yh) < minsize) or \
           (countwhat == "unique" and len(np.unique(yh)) < minsize):
            bigenough = False
            
    if verbose:
        print(f"Unconstrained objective = {obj:.6f}")
        print("Initial cluster sizes:", dict(Counter(clusvec)))

    #---------------------------------------------------------------------------------------------
    # --- Step 2: Iterative reassignment with OT
    #---------------------------------------------------------------------------------------------
    converged = False
    it = 0
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True)
    def solve_transport(cost, nrows, ncols, minsize):
        """
        Solve the transportation problem with constraints:
        - each row i must sum to 1
        - each column (demand) must sum to at least minsize
        Integer constraints ensure atoms cannot be split. (Each obs must be 
        assigned entirely to one cluster, not fractionally across several)
        """
        prob = pulp.LpProblem("transport", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Binary")
              for j in range(ncols)] for i in range(nrows)]

        # objective
        prob += pulp.lpSum(cost[i, j] * x[i][j] for i in range(nrows) for j in range(ncols))
        # row constraints
        for i in range(nrows):
            prob += pulp.lpSum(x[i][j] for j in range(ncols)) == 1

        for j in range(ncols):
            prob += pulp.lpSum(x[i][j] for i in range(nrows)) >= minsize
            
        prob.solve(solver)
        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError("LP solver failed")
        sol = np.array([[pulp.value(x[i][j]) for j in range(ncols)] for i in range(nrows)])
        return sol
        
    num = 0
    if bigenough == False:
        if countwhat == "any":
            num = 1
            # Stop when maxit reached or when no assignment changes
            while it < maxit and num > 0:
                if verbose:
                    print("\nIteration Number = ", it,"\n")
                    
                #y_sorted[:, None] of shape:  (n,1) and centers[None, :]:  (k,1)
                #--> result (n,k) with cost[i, j] = |y_sorted[i] - centers[j]|         
                cost = np.abs(y_sorted[:, None] - centers[None, :])
                sol = solve_transport(cost, n, k, minsize) #sol[i, j] ≈ 1 if observation i is assigned to cluster j, and 0 otherwise
                clusvecnew = np.argmax(sol, axis=1) #finds the index of the maximum value along each row
    
                num = np.sum(clusvecnew != clusvec) #noneed to be invariant to labels name
                if verbose:
                    print(f"Number of assignment changes = {num}")
                if num > 0:
                    clusvec = clusvecnew
                    centers = np.array([np.median(y_sorted[clusvec == h]) for h in range(k)])
                    obj = objL1(y_sorted, centers, clusvec)
                    objectives.append(obj)
                    it += 1
                    if verbose:
                        print("objective = ", obj, "\n")
                        print(clusvec)
                        print(f"Iteration {it}: sizes={dict(Counter(clusvec))}, centers={centers}")
    
        if countwhat == "unique":
    
            yuniq, whichuni, inv = np.unique(y_sorted, return_index=True, return_inverse=True)
            # e.g.,: y_sorted = [1, 1, 2, 2, 2, 5]
            # yuniq   = [1, 2, 5]
            # whichuni= [0, 2, 5]
            # inv = [0, 0, 1, 1, 1, 2]  # tells you each element’s unique index
    
            clusuni = clusvec[whichuni]
            num = 1
            while it < maxit and num > 0:
                if verbose: 
                    print("\nIteration Number = ", it, "\n")
                
                #similar logic than when countwhat == "any"
                cost = np.abs(yuniq[:, None] - centers[None, :])
                sol = solve_transport(cost, nuniq, k, minsize) 
                clusuninew = np.argmax(sol, axis=1)
                num = np.sum(clusuninew != clusuni)
                if verbose: 
                    print(f"Number of assignment changes = {num}")
                if num > 0:
                    clusuni = clusuninew
                    clusvec = clusuni[inv] #include the cluster for the duplicates
                    centers = np.array([np.median(y_sorted[clusvec == h]) for h in range(k)])
                    obj = objL1(y_sorted, centers, clusvec)
                    objectives.append(obj)
                    it += 1
                    if verbose: 
                        print("objective = ", obj, "\n")
                        print(clusvec)
                        print(f"Iteration {it}: sizes={dict(Counter(clusvec))}, centers={centers}")
    
    #---------------------------------------------------------------------------------------------
    # --- Postprocessing: restore original order and NAs
    #---------------------------------------------------------------------------------------------
    if verbose:
        print('clusterID on sorted y: ',clusvec)
        
    # unsort back to valid_idx order
    clus_unsorted = np.empty(n, dtype=float)
    clus_unsorted[sort_idx] = clusvec

    # insert into full-length vector
    clustering_full = np.full(n_total, np.nan)
    clustering_full[valid_idx] = clus_unsorted

    return {
        "iter": it,
        "converged": num==0,
        "clustering": clustering_full,
        "centers": centers,
        "sizes": dict(Counter(clusvec)),
        "objective": objL1(y_sorted, centers, clusvec),
        "objectives": objectives
    }



    