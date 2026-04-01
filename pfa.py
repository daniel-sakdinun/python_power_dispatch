import numpy as np

class Bus:
    def __init__(self, v: float | None, phi: float | None, P: float | None, Q: float | None):
        self.v = v
        self.phi = phi
        self.P = P
        self.Q = Q
        
        if self.v is not None and self.phi is not None and self.P is None and self.Q is None: 
            self.type = 'Slack'
        elif self.v is not None and self.phi is None and self.P is not None and self.Q is None: 
            self.type = 'PV'
        elif self.v is None and self.phi is None and self.P is not None and self.Q is not None: 
            self.type = 'PQ'
        else: 
            self.type = 'None'
    
    def get(self) -> list:
        return [self.v, self.phi, self.P, self.Q]
    
    def flat_start(self):
        if self.type == 'PV': 
            self.phi = 0.0
        elif self.type == 'PQ':
            self.v = 1.0
            self.phi = 0.0
    
    def reset(self):
        if self.type == 'PV': 
            self.phi = None
        elif self.type == 'PQ':
            self.v = None
            self.phi = None

def calculate_P_Q(busses, Ybus, i):
    P_i = 0.0
    Q_i = 0.0
    V_i = busses[i].v
    phi_i = busses[i].phi
    G = Ybus.real
    B = Ybus.imag
    
    for j in range(len(busses)):
        V_j = busses[j].v
        phi_j = busses[j].phi
        delta_ij = phi_i - phi_j
        
        # P_i = sum( |Vi||Vj|(Gij*cos(d_ij) + Bij*sin(d_ij)) )
        P_i += V_i * V_j * (G[i, j] * np.cos(delta_ij) + B[i, j] * np.sin(delta_ij))
        # Q_i = sum( |Vi||Vj|(Gij*sin(d_ij) - Bij*cos(d_ij)) )
        Q_i += V_i * V_j * (G[i, j] * np.sin(delta_ij) - B[i, j] * np.cos(delta_ij))
        
    return P_i, Q_i

def jacobian(busses, Ybus):
    G = Ybus.real
    B = Ybus.imag
    
    unknown_delta_idx = [i for i, bus in enumerate(busses) if bus.type in ['PV', 'PQ']]
    unknown_v_idx = [i for i, bus in enumerate(busses) if bus.type == 'PQ']
    
    n_delta = len(unknown_delta_idx)
    n_v = len(unknown_v_idx)
    size = n_delta + n_v
    
    J = np.zeros((size, size))
    
    for r, i in enumerate(unknown_delta_idx):
        P_i, Q_i = calculate_P_Q(busses, Ybus, i)
        V_i = busses[i].v
        
        for c, j in enumerate(unknown_delta_idx):
            V_j = busses[j].v
            delta_ij = busses[i].phi - busses[j].phi
            
            if i != j: # Off-diagonal
                J[r, c] = V_i * V_j * (G[i, j] * np.sin(delta_ij) - B[i, j] * np.cos(delta_ij))
            else:      # Diagonal
                J[r, c] = -Q_i - B[i, i] * (V_i**2)
                
        for c, j in enumerate(unknown_v_idx):
            col = n_delta + c
            V_j = busses[j].v
            delta_ij = busses[i].phi - busses[j].phi
            
            if i != j: # Off-diagonal
                J[r, col] = V_i * (G[i, j] * np.cos(delta_ij) + B[i, j] * np.sin(delta_ij))
            else:      # Diagonal
                J[r, col] = (P_i / V_i) + G[i, i] * V_i

    for r, i in enumerate(unknown_v_idx):
        row = n_delta + r # ขยับแถวลงมาด้านล่างต่อจากสมการ P
        P_i, Q_i = calculate_P_Q(busses, Ybus, i)
        V_i = busses[i].v
        
        # ส่วน J3: dQ_i / d_delta_j
        for c, j in enumerate(unknown_delta_idx):
            V_j = busses[j].v
            delta_ij = busses[i].phi - busses[j].phi
            
            if i != j: # Off-diagonal
                J[row, c] = -V_i * V_j * (G[i, j] * np.cos(delta_ij) + B[i, j] * np.sin(delta_ij))
            else:      # Diagonal
                J[row, c] = P_i - G[i, i] * (V_i**2)
                
        for c, j in enumerate(unknown_v_idx):
            col = n_delta + c
            V_j = busses[j].v
            delta_ij = busses[i].phi - busses[j].phi
            
            if i != j: # Off-diagonal
                J[row, col] = V_i * (G[i, j] * np.sin(delta_ij) - B[i, j] * np.cos(delta_ij))
            else:      # Diagonal
                J[row, col] = (Q_i / V_i) - B[i, i] * V_i

    return J

def mismatch(busses, Ybus):
    delta_P = []
    delta_Q = []
    
    for i, bus in enumerate(busses):
        if bus.type == 'Slack':
            continue
            
        P_calc, Q_calc = calculate_P_Q(busses, Ybus, i)
        
        if bus.type in ['PV', 'PQ']:
            delta_P.append(bus.P - P_calc)
            
        if bus.type == 'PQ':
            delta_Q.append(bus.Q - Q_calc)
            
    mismatch = np.array(delta_P + delta_Q).reshape(-1, 1)
    
    return mismatch

busses = [
    # v (pu), phi, P (pu), Q (pu)
    Bus(1.0, 0.0, None, None),       # Bus 1: Slack
    Bus(1.5, None, 1.0, None),       # Bus 2: PV (P=100MW)
    Bus(1.5, None, 1.0, None),       # Bus 3: PV (P=100MW)
    Bus(None, None, -1.8, -0.5),     # Bus 4: PQ (Load P=180MW, Q=50MVAR)
]

Ybus = 1j * np.array([
    [-8.5, 2.5, 5, 0],
    [2.5, -8.75, 5, 0],
    [5, 5, -22.5, 12.5],
    [0, 0, 12.5, -12.5]
])

for bus in busses:
    bus.flat_start()

J = jacobian(busses, Ybus)
m = mismatch(busses, Ybus)
correction = np.linalg.solve(J, m)

print()