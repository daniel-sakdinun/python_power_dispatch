import numpy as np
from edp import Grid, Generator
from loss import DCFlowLoss
from eds import newton

R_lines = np.array([0.02, 0.015, 0.03, 0.025]) 

PTDF_matrix = np.array([
    [ 0.4, -0.2,  0.1],
    [ 0.6,  0.3, -0.1],
    [-0.5,  0.8,  0.2],
    [ 0.1,  0.5,  0.9]
])

dc_loss = DCFlowLoss(num_gen=3, PTDF=PTDF_matrix, R_lines=R_lines)

grid = Grid(
    800,
    [
        Generator(500, 5.3, .004, 200, 450),
        Generator(400, 5.5, .006, 150, 350),
        Generator(200, 5.8, .009, 100, 250)
    ],
    loss=dc_loss
)

grid = newton(grid)

print("=== Results ===")
print(f"Total cost: {grid.TC():.2f} $/h")
print(f"Total loss: {grid.loss.Pl():.4f} MW")
print(f"Marginal Cost: {grid.MC:.4f} $/MWh")
for i, gen in enumerate(grid.G):
    print(f"Gen {i+1} Power: {gen.P:.2f} MW")