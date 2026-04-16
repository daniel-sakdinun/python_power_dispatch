import yaml
import numpy as np
import cvxpy as cp
import networkx as nx
import scipy.optimize as opt

from pathlib import Path
from fourpace.model import BusComponent, SynchronousMachine, AsynchronousMachine, Load, Shunt, Inverter, Battery, BranchComponent, TransmissionLine, Transformer

class Bus(nx.Graph):
    def __init__(self, name: str, Vbase: float, type: str = 'PQ'):
        super().__init__()
        self.name = name
        self.Vbase: float | None = Vbase
        
        self.type = type
        self.V:float = 1.0
        self.theta:float = 0.0
        self.add_node(name, obj=self)
    
    @property
    def P(self) -> float:
        total_p = 0.0
        for comp in self.components:
            if hasattr(comp, 'P'):
                total_p += comp.P
        return total_p

    @property
    def Q(self) -> float:
        total_q = 0.0
        for comp in self.components:
            if hasattr(comp, 'Q'):
                total_q += comp.Q
        return total_q
    
    @property
    def S(self) -> complex:
        return self.P + 1j*self.Q;
    
    def component(self, name: str) -> BusComponent:
        if name in self.nodes:
            return self.nodes[name]['obj']
        
        raise KeyError(f"❌ '{name}' not found in this bus.")
    
    @property
    def components(self) -> list:
        comp_list = []
        for _, data in self.nodes(data=True):
            obj = data.get('obj')
            if obj is not None and obj is not self:
                comp_list.append(obj)
        return comp_list
    
    def get(self):
        return [self.V, self.theta, self.P, self.Q]
    
    def add_component(self, component: BusComponent):
        self.add_node(component.name, obj=component)
        self.add_edge(self.name, component.name)
    
    def add_components(self, components: list[BusComponent]):
        for component in components:
            self.add_component(component)
        
    def total_cost(self) -> float:
        components = [component for _, component in self.nodes(data='obj')]
        components.remove(self)
        sum = 0
        for com in components:
            sum += com.cost()
        return sum

class Grid(nx.Graph):
    def __init__(self, Sbase: float):
        super().__init__()
        self.Ybus: np.ndarray | None = None
        self.Sbase: float = Sbase
    
    @classmethod
    def load(cls, filepath: str) -> 'Grid':
        path = Path(filepath)
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError("❌ Unsupported format! Please use .yaml or .json")

        grid = cls(Sbase=data.get('Sbase', 100.0))

        comp_classes = {
            'SynchronousMachine': SynchronousMachine,
            'AsynchronousMachine': AsynchronousMachine,
            'Load': Load,
            'Shunt': Shunt,
            'Inverter': Inverter,
            'Battery': Battery,
            'TransmissionLine': TransmissionLine,
            'Transformer': Transformer
        }

        for b_data in data.get('buses', []):
            bus = Bus(name=b_data['name'], Vbase=b_data['Vbase'], type=b_data.get('bus_type', 'PQ'))
            grid.add_bus(bus)
            
            for comp_data in b_data.get('components', []):
                comp_type = comp_data.pop('type')
                if comp_type in comp_classes:
                    comp_obj = comp_classes[comp_type](**comp_data) 
                    grid.bus(bus.name).add_component(comp_obj)

        for branch_data in data.get('branches', []):
            branch_type = branch_data.pop('type')
            from_bus = branch_data.pop('from_bus')
            to_bus = branch_data.pop('to_bus')
            
            if branch_type in comp_classes:
                branch_obj = comp_classes[branch_type](**branch_data)
                grid.connect(from_bus, to_bus, branch_obj)

        print(f"✅ Successfully loaded grid from {path.name}")
        return grid
    
    def add_bus(self, bus:Bus):
        self.add_node(bus.name, bus=bus)
    
    def add_busses(self, busses:list[Bus]):
        for bus in busses:
            self.add_bus(bus)
        
    def connect(self, from_bus:str, to_bus:str, branch: BranchComponent):
        self.add_edge(from_bus, to_bus, obj=branch)
    
    def bus(self, name: str) -> Bus:
        if name in self.nodes:
            return self.nodes[name]['bus']
        
        raise KeyError(f"❌  Bus  '{name}' not found in this grid.")
    
    @property
    def buses(self) -> list[Bus]:
        return [bus for _, bus in self.nodes(data='bus')]
    
    def build_ybus(self) -> np.ndarray:
        nodes = list(self.nodes)
        n = len(nodes)
        Y = np.zeros((n, n), dtype=complex)
        
        for u, v, data in self.edges(data=True):
            i = nodes.index(u)
            j = nodes.index(v)
            obj = data.get('obj')
            
            if obj is None: continue
            
            y = obj.Y
            
            if isinstance(obj, TransmissionLine):
                b_sh = obj.B_shunt
                Y[i, i] += y + (1j * b_sh / 2)
                Y[j, j] += y + (1j * b_sh / 2)
                Y[i, j] -= y
                Y[j, i] -= y
                
            elif isinstance(obj, Transformer):
                a = obj.tap_ratio
                alpha = obj.phase_shift
                
                tap_complex = a * np.exp(1j * alpha)
                
                Y[i, i] += y / (a**2)
                Y[j, j] += y
                Y[i, j] -= y / np.conj(tap_complex)
                Y[j, i] -= y / tap_complex
                
        self.Ybus:np.ndarray = Y
        return Y
    
    def result(self):
        print("\n=== 📊 POWER FLOW RESULTS ===")
        for i, bus in enumerate(self.buses):
            P_pu, Q_pu = self.calculate_PQ(i)
            
            P_actual:float = P_pu * self.Sbase
            Q_actual:float = Q_pu * self.Sbase
            
            print(f"Bus {bus.name} | V = {bus.V:.4f} pu | phase = {np.rad2deg(bus.theta):.2f}° | P = {P_actual:7.2f} MW | Q = {Q_actual:7.2f} MVAr")
    
    def loading_status(self):
        grid_status = self.check_overload()
        
        print("\n📊 Grid Loading Status:")
        print("--- Branches ---")
        for name, pct in grid_status['branches'].items():
            status = "🔥 OVERLOAD!" if pct > 100 else "✅ OK"
            print(f"{name}: {pct}% {status}")

        print("\n--- Generators ---")
        for name, pct in grid_status['generators'].items():
            status = "🔥 OVERLOAD!" if pct > 100 else "✅ OK"
            print(f"{name}: {pct}% {status}")
    
    def apply_profile(self, multipliers: dict):
        for bus in self.buses:
            for comp in bus.components:
                if comp.name in multipliers:
                    m = multipliers[comp.name]
                    
                    if isinstance(comp, Load):
                        comp.P = comp.P_nom * m
                        comp.Q = comp.Q_nom * m
                        
                    elif isinstance(comp, AsynchronousMachine):
                        comp.P_rated = comp.P_rated_base * m 
                        
                    elif isinstance(comp, Shunt):
                        comp.Q_nom = comp.Q_nom_base * m
                        
                    elif isinstance(comp, Inverter):
                        comp.P = comp.S_max_base * m
    
    def calculate_PQ(self, i) -> tuple[float, float]:
        bus_i:Bus = self.buses[i]
        Vi = bus_i.V
        theta_i = bus_i.theta

        G:np.ndarray = self.Ybus.real
        B:np.ndarray = self.Ybus.imag

        P_i:float = 0.0
        Q_i:float = 0.0

        for j in range(len(self.buses)):
            bus_j:Bus = self.buses[j]
            Vj:float = bus_j.V
            theta_j:float = bus_j.theta
            delta_ij:float = theta_i - theta_j

            P_i += Vi * Vj * (G[i, j] * np.cos(delta_ij) + B[i, j] * np.sin(delta_ij))
            Q_i += Vi * Vj * (G[i, j] * np.sin(delta_ij) - B[i, j] * np.cos(delta_ij))
        
        return P_i, Q_i
    
    def update_motor_slip(self, motor, V_pu):
        def torque_mismatch(s):
            if s <= 0: return 1e6
            
            V_phase = (V_pu * motor.V_rated) / np.sqrt(3)
            ws = 2 * np.pi * motor.freq / (motor.poles / 2)
            
            R_total = motor.Rs + (motor.Rr / s)
            X_total = motor.Xs + motor.Xr
            Z_sq = R_total**2 + X_total**2
            
            Te = (3 * V_phase**2 * (motor.Rr / s)) / (ws * Z_sq)
            
            if motor.load_type == 'constant_torque':
                Tm = (motor.P_rated * 1000) / ws 
            else:
                Tm = ((motor.P_rated * 1000) / ws) * ((1 - s)**2)
                
            return Te - Tm

        new_s = opt.fsolve(torque_mismatch, x0=motor.s)[0]
        motor.s = max(0.0001, min(new_s, 0.99))
    
    def check_overload(self) -> dict:
        loadings = {
            'branches': {},
            'generators': {}
        }
        
        # ==========================================
        # 1. Branches Check
        # ==========================================
        for u, v, data in self.edges(data=True):
            branch = data.get('obj')
            if branch is None or getattr(branch, 'S_max', None) is None:
                continue
                
            from_bus = self.bus(u)
            to_bus = self.bus(v)
            
            V_i = from_bus.V * np.exp(1j * from_bus.theta)
            V_j = to_bus.V * np.exp(1j * to_bus.theta)
            
            # Calculate Line Current (exclude OLTC Transformer)
            if type(branch).__name__ == 'Transformer':
                t = branch.tap_ratio * np.exp(1j * branch.phase_shift)
                I_ij = (V_i/t - V_j) * branch.Y
            else:
                I_ij = (V_i - V_j) * branch.Y
                
            # Calculate Apparent Power (S = V * I*) in MVA
            S_flow_MVA = abs(V_i * np.conj(I_ij)) * self.Sbase
            
            # Percentage
            loading_pct = (S_flow_MVA / branch.S_max) * 100
            loadings['branches'][branch.name] = round(loading_pct, 2)

        # ==========================================
        # 2. Generator Check
        # ==========================================
        for bus in self.buses:
            for comp in bus.components:
                if type(comp).__name__ == 'SynchronousMachine':
                    S_gen_MVA = np.sqrt(comp.P**2 + comp.Q**2)
                    
                    s_rated = getattr(comp, 'S_rated', None)
                    
                    if s_rated is None:
                        p_max = getattr(comp, 'Pmax', 9999.0)
                        q_max = getattr(comp, 'Qmax', 9999.0)
                        
                        if p_max == float('inf') or q_max == float('inf') or p_max == 9999.0:
                            s_rated = 9999.0
                        else:
                            s_rated = np.sqrt(p_max**2 + q_max**2)
                            
                    if s_rated > 0 and s_rated != 9999.0:
                        loading_pct = (S_gen_MVA / s_rated) * 100
                        loadings['generators'][comp.name] = round(loading_pct, 2)
                    else:
                        loadings['generators'][comp.name] = 0.0
                        
        return loadings

import numpy as np
import cvxpy as cp

def CEP(grid, profile_df, relax: str = 'SOCP', solver: str = 'SCS'):
    """
    Capacity Expansion Planning (CEP) Engine.
    Optimizes both investment costs (CapEx) and operating costs (OpEx) 
    to determine the optimal sizing of candidate components (Batteries & Solar).
    """
    print("\n🏗️ Starting Capacity Expansion Planning (CEP)...")
    
    T = len(profile_df)
    num_bus = len(grid.buses)
    edges = list(grid.edges(data=True))
    num_branch = len(edges)
    nodes_list = list(grid.nodes)

    # 1. Device Classification
    machines, inverters, batteries = [], [], []
    machine_bus, inverter_bus, battery_bus = [], [], []

    for i, bus in enumerate(grid.buses):
        for comp in bus.components:
            c_name = type(comp).__name__
            if c_name == 'SynchronousMachine': machines.append(comp); machine_bus.append(i)
            elif c_name == 'Inverter': inverters.append(comp); inverter_bus.append(i)
            elif c_name == 'Battery': batteries.append(comp); battery_bus.append(i)

    num_gen, num_inv, num_bat = len(machines), len(inverters), len(batteries)

    # 2. โหลดข้อมูล Load Profile และ Solar Data 24 ชม.
    P_load_pu = np.zeros((num_bus, T))
    Q_load_pu = np.zeros((num_bus, T))
    Q_shunt_pu = np.zeros((num_bus, T))
    P_solar_avail_pu = np.zeros((num_inv, T)) if num_inv > 0 else None

    for t in range(T):
        grid.apply_profile(profile_df.iloc[t].to_dict())
        for i, bus in enumerate(grid.buses):
            for comp in bus.components:
                c_name = type(comp).__name__
                if c_name == 'Load':
                    P_load_pu[i, t] += abs(comp.P) / grid.Sbase
                    Q_load_pu[i, t] += abs(comp.Q) / grid.Sbase
                elif c_name == 'AsynchronousMachine':
                    comp.update_pq_from_slip(1.0, grid.Sbase)
                    P_load_pu[i, t] += abs(comp.P) / grid.Sbase
                    Q_load_pu[i, t] += abs(comp.Q) / grid.Sbase
                elif c_name == 'Shunt':
                    Q_shunt_pu[i, t] += comp.Q_nom / grid.Sbase
                elif c_name == 'Inverter':
                    inv_idx = inverters.index(comp)
                    # 🌟 FIX: ถ้าเป็น Candidate ดึงค่าทศนิยม(แดด)จากโปรไฟล์เพียวๆ
                    if getattr(comp, 'is_candidate', False):
                        multiplier = profile_df.iloc[t].to_dict().get(comp.name, 0.0)
                        P_solar_avail_pu[inv_idx, t] = multiplier
                    else:
                        P_solar_avail_pu[inv_idx, t] = comp.P / grid.Sbase

    grid.build_ybus()
    G_bus = grid.Ybus.real
    B_bus = grid.Ybus.imag
    
    # ====================================================================
    # 🌟 3. SHARED VARIABLES (ตัวแปรที่ใช้ร่วมกัน 24 ชม.)
    # ====================================================================
    Pg = cp.Variable((num_gen, T)) if num_gen > 0 else None
    Qg = cp.Variable((num_gen, T)) if num_gen > 0 else None
    P_inv = cp.Variable((num_inv, T)) if num_inv > 0 else None
    Q_inv = cp.Variable((num_inv, T)) if num_inv > 0 else None
    P_ch = cp.Variable((num_bat, T)) if num_bat > 0 else None
    P_dis = cp.Variable((num_bat, T)) if num_bat > 0 else None
    
    # 🌟 FIX: เปลี่ยน SoC เป็น E_stored (เก็บค่าพลังงานจริง MWh เพื่อเลี่ยงการหารด้วยตัวแปร)
    E_stored = cp.Variable((num_bat, T)) if num_bat > 0 else None

    constraints = []
    slack_idx = next(idx for idx, b in enumerate(grid.buses) if b.type == 'Slack')

    # ====================================================================
    # 🌟 4. NETWORK RELAXATION (SOCP / SDP)
    # ====================================================================
    if relax.upper() == 'SOCP':
        # --- BRANCH FLOW MODEL ---
        v = cp.Variable((num_bus, T))
        P_ij = cp.Variable((num_branch, T))
        Q_ij = cp.Variable((num_branch, T))
        l_ij = cp.Variable((num_branch, T))

        for t in range(T):
            for i in range(num_bus):
                constraints.append(v[i, t] == 1.0 if i == slack_idx else v[i, t] >= 0.85**2)
                if i != slack_idx: constraints.append(v[i, t] <= 1.15**2)

            P_inj = {i: 0.0 for i in range(num_bus)}; Q_inj = {i: 0.0 for i in range(num_bus)}

            for k, (u, target_v, data) in enumerate(edges):
                i, j = nodes_list.index(u), nodes_list.index(target_v)
                branch = data.get('obj')
                R, X, Z2 = branch.R, branch.X, branch.R**2 + branch.X**2
                tau = branch.tap_ratio if type(branch).__name__ == 'Transformer' else 1.0
                v_i_eff = v[i, t] / (tau**2)

                constraints.append(v[j, t] == v_i_eff - 2*(R*P_ij[k, t] + X*Q_ij[k, t]) + Z2*l_ij[k, t])
                constraints.append(cp.SOC(v_i_eff + l_ij[k, t], cp.vstack([2*P_ij[k, t], 2*Q_ij[k, t], v_i_eff - l_ij[k, t]])))

                if getattr(branch, 'S_max', None):
                    constraints.append(cp.norm(cp.vstack([P_ij[k, t], Q_ij[k, t]])) <= branch.S_max / grid.Sbase)

                P_inj[i] += P_ij[k, t]; Q_inj[i] += Q_ij[k, t]
                P_inj[j] -= (P_ij[k, t] - R*l_ij[k, t]); Q_inj[j] -= (Q_ij[k, t] - X*l_ij[k, t])

            for i in range(num_bus):
                p_g = sum([Pg[k, t] for k, b_idx in enumerate(machine_bus) if b_idx == i]) if num_gen > 0 else 0.0
                q_g = sum([Qg[k, t] for k, b_idx in enumerate(machine_bus) if b_idx == i]) if num_gen > 0 else 0.0
                p_i = sum([P_inv[k, t] for k, b_idx in enumerate(inverter_bus) if b_idx == i]) if num_inv > 0 else 0.0
                q_i = sum([Q_inv[k, t] for k, b_idx in enumerate(inverter_bus) if b_idx == i]) if num_inv > 0 else 0.0
                p_d = sum([P_dis[k, t] for k, b_idx in enumerate(battery_bus) if b_idx == i]) if num_bat > 0 else 0.0
                p_c = sum([P_ch[k, t] for k, b_idx in enumerate(battery_bus) if b_idx == i]) if num_bat > 0 else 0.0
                constraints.append(p_g + p_i + p_d - p_c - P_load_pu[i, t] == P_inj[i])
                constraints.append(q_g + q_i - Q_load_pu[i, t] + (Q_shunt_pu[i, t] * v[i, t]) == Q_inj[i])

    elif relax.upper() == 'SDP':
        # --- BUS INJECTION MODEL (HERMITIAN MATRIX W) ---
        W = [cp.Variable((num_bus, num_bus), hermitian=True) for _ in range(T)]
        for t in range(T):
            constraints.append(W[t] >> 0)
            WR, WI = cp.real(W[t]), cp.imag(W[t])
            for i in range(num_bus):
                constraints.append(WR[i, i] == 1.0 if i == slack_idx else WR[i, i] >= 0.85**2)
                if i != slack_idx: constraints.append(WR[i, i] <= 1.15**2)

                P_calc = G_bus[i, :] @ WR[i, :] + B_bus[i, :] @ WI[i, :]
                Q_calc = G_bus[i, :] @ WI[i, :] - B_bus[i, :] @ WR[i, :]

                p_g = sum([Pg[k, t] for k, b_idx in enumerate(machine_bus) if b_idx == i]) if num_gen > 0 else 0.0
                q_g = sum([Qg[k, t] for k, b_idx in enumerate(machine_bus) if b_idx == i]) if num_gen > 0 else 0.0
                p_i = sum([P_inv[k, t] for k, b_idx in enumerate(inverter_bus) if b_idx == i]) if num_inv > 0 else 0.0
                q_i = sum([Q_inv[k, t] for k, b_idx in enumerate(inverter_bus) if b_idx == i]) if num_inv > 0 else 0.0
                p_d = sum([P_dis[k, t] for k, b_idx in enumerate(battery_bus) if b_idx == i]) if num_bat > 0 else 0.0
                p_c = sum([P_ch[k, t] for k, b_idx in enumerate(battery_bus) if b_idx == i]) if num_bat > 0 else 0.0
                constraints.append(P_calc == p_g + p_i + p_d - p_c - P_load_pu[i, t])
                constraints.append(Q_calc == q_g + q_i - Q_load_pu[i, t] + (Q_shunt_pu[i, t] * WR[i, i]))

    # ====================================================================
    # 🌟 5. CEP SPECIFIC: INVESTMENT VARIABLES & CONSTRAINTS
    # ====================================================================
    total_capex = 0.0
    
    bat_p_max_capacity = []
    bat_e_max_capacity = []
    inv_s_max_capacity = [] # 🌟 เพิ่มสำหรับ Solar Candidate

    # --- พิจารณาลงทุนสร้าง Solar ---
    if num_inv > 0:
        for k, inv in enumerate(inverters):
            if getattr(inv, 'is_candidate', False):
                built_s_mw = cp.Variable(nonneg=True)
                constraints.append(built_s_mw <= inv.max_build_mw)
                
                inv_s_max_capacity.append(built_s_mw / grid.Sbase)
                total_capex += (built_s_mw * inv.capex_per_mw) * getattr(inv, 'daily_capex_factor', 1.0)
            else:
                inv_s_max_capacity.append(inv.S_max / grid.Sbase)

    # --- พิจารณาลงทุนสร้าง Battery ---
    if num_bat > 0:
        for k, bat in enumerate(batteries):
            if getattr(bat, 'is_candidate', False):
                built_p_mw = cp.Variable(nonneg=True)
                built_e_mwh = cp.Variable(nonneg=True)
                
                constraints.extend([
                    built_p_mw <= bat.max_build_mw,
                    built_e_mwh <= bat.max_build_mwh
                ])
                
                bat_p_max_capacity.append(built_p_mw / grid.Sbase)
                bat_e_max_capacity.append(built_e_mwh / grid.Sbase)
                total_capex += (built_p_mw * bat.capex_per_mw + built_e_mwh * bat.capex_per_mwh) * getattr(bat, 'daily_capex_factor', 1.0)
            else:
                bat_p_max_capacity.append(bat.P_max / grid.Sbase)
                bat_e_max_capacity.append(bat.E_max / grid.Sbase)

    # ====================================================================
    # 🌟 6. TIME-COUPLING WITH DYNAMIC CAPACITY
    # ====================================================================
    for t in range(T):
        # --- Generator Constraints ---
        if num_gen > 0:
            for k, m in enumerate(machines):
                constraints.extend([
                    Pg[k, t] >= (m.Pmin if m.Pmin != float('-inf') else 0.0) / grid.Sbase,
                    Pg[k, t] <= (m.Pmax if m.Pmax != float('inf') else 9999.0) / grid.Sbase,
                    Qg[k, t] >= (m.Qmin if m.Qmin != float('-inf') else -9999.0) / grid.Sbase,
                    Qg[k, t] <= (m.Qmax if m.Qmax != float('inf') else 9999.0) / grid.Sbase
                ])

        # --- Inverter Constraints ---
        if num_inv > 0:
            for k, inv in enumerate(inverters):
                if getattr(inv, 'is_candidate', False):
                    p_avail = P_solar_avail_pu[k, t] * inv_s_max_capacity[k]
                else:
                    p_avail = P_solar_avail_pu[k, t]

                constraints.extend([
                    P_inv[k, t] >= 0,
                    P_inv[k, t] <= p_avail,
                    cp.norm(cp.vstack([P_inv[k, t], Q_inv[k, t]])) <= inv_s_max_capacity[k]
                ])

        # --- Battery Constraints ---
        if num_bat > 0:
            for k, bat in enumerate(batteries):
                p_max_pu = bat_p_max_capacity[k]
                e_max_pu = bat_e_max_capacity[k]
                
                constraints.extend([
                    P_ch[k, t] >= 0, P_ch[k, t] <= p_max_pu,
                    P_dis[k, t] >= 0, P_dis[k, t] <= p_max_pu,
                    E_stored[k, t] >= 0.1 * e_max_pu,
                    E_stored[k, t] <= 1.0 * e_max_pu
                ])

                delta_e = (P_ch[k, t] * bat.eta - P_dis[k, t] / bat.eta)
                if t == 0:
                    init_e = bat.init_soc * e_max_pu
                    constraints.append(E_stored[k, t] == init_e + delta_e)
                else:
                    constraints.append(E_stored[k, t] == E_stored[k, t-1] + delta_e)

    # ====================================================================
    # 🌟 7. OBJECTIVE: MINIMIZE(CAPEX + OPEX)
    # ====================================================================
    total_opex = 0
    for t in range(T):
        if num_gen > 0:
            for k, m in enumerate(machines):
                Pg_mw = Pg[k, t] * grid.Sbase
                total_opex += m.a + (m.b * Pg_mw) + (m.c * cp.square(Pg_mw))
    
    total_cost = total_capex + total_opex

    prob = cp.Problem(cp.Minimize(total_cost), constraints)
    print(f"⏳ Optimizing Sizing and Operation ({relax.upper()})...")
    
    try:
        prob.solve(solver=getattr(cp, solver.upper()), verbose=False)
    except Exception as e:
        print(f"❌ CVXPY Error: {e}")

    if prob.status in ["optimal", "optimal_inaccurate"]:
        print(f"✅ CEP Converged! Total Cost (CapEx + OpEx): ${prob.value:.2f}")
        
        # --- พิมพ์ผลการตัดสินใจสร้าง Solar ---
        if num_inv > 0:
            for k, inv in enumerate(inverters):
                if getattr(inv, 'is_candidate', False):
                    inv.built_S_max = float(inv_s_max_capacity[k].value * grid.Sbase)
                    print(f"   ☀️ Investment Decision -> {inv.name}: S_max = {inv.built_S_max:.2f} MW")

        # --- พิมพ์ผลการตัดสินใจสร้าง Battery ---
        if num_bat > 0:
            for k, bat in enumerate(batteries):
                if getattr(bat, 'is_candidate', False):
                    bat.built_P_max = float(bat_p_max_capacity[k].value * grid.Sbase)
                    bat.built_E_max = float(bat_e_max_capacity[k].value * grid.Sbase)
                    print(f"   🔋 Investment Decision -> {bat.name}: P_max = {bat.built_P_max:.2f} MW, E_max = {bat.built_E_max:.2f} MWh")
    else:
        raise Exception(f"❌ CEP Infeasible! Status: {prob.status}")