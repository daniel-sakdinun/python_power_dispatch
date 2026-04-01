import numpy as np
from edp import Grid

def newton(grid: Grid, tolerance=1e-4, max_iteration=100):
    free_idx = grid.free_idx
    
    c = np.array([gen.c for gen in grid.NLG])
        
    for iteration in range(max_iteration):
        grid.loss.Pg = np.array([gen.P for gen in grid.G])
        
        grad = grid.loss.gradient()
        penalty_full = 1 - grad
        hessian_full = grid.loss.hessian()
        
        penalty_free = penalty_full[free_idx]
        hessian_free = hessian_full[np.ix_(free_idx, free_idx)]
        
        top_left = np.diag(2 * c) + (grid.MC * hessian_free)
        top_right = -penalty_free.reshape(-1, 1)
        bottom_left = -penalty_free.reshape(1, -1)
        
        J = np.block([
            [top_left, top_right],
            [bottom_left, 0]
        ])
        
        mismatch = []
        for i in free_idx:
            gen = grid.G[i]
            # F_i = b_i + 2c_i*P_i - lambda*(1 - dPl/dPi)
            mismatch.append(gen.b + 2*gen.c*gen.P - grid.MC * penalty_full[i])
            
        # dP = Pd + Pl - sum(Pg)
        mismatch.append(grid.Pd + grid.loss.Pl() - sum(grid.Pg()))
        mismatch = np.array(mismatch)
            
        if np.max(np.abs(mismatch)) < tolerance:
            print(f"Converged at iteration: {iteration}!\n")
            break

        correction = np.linalg.solve(J, -mismatch)

        for i, free_index in enumerate(free_idx):
            grid.G[free_index].P += correction[i]

        grid.MC += correction[-1]
    
    while grid.KKT():
        if len(grid.NLG) == 0:
            print("\n🚨 CRITICAL ERROR: System Out of Capacity!\n")
            break
        
        newton(grid)

    return grid