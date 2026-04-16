import json
import numpy as np
import cvxpy as cp

from pandas.core.frame import DataFrame
from fourpace.model import SynchronousMachine, AsynchronousMachine, Load, Shunt, Transformer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item() 
        
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
            
        return super(NumpyEncoder, self).default(obj)

def MPOPF(grid, profile_df, relax: str = 'SOCP', solver_name: str = 'SCS'):
    T = len(profile_df)
    num_bus = len(grid.buses)
    edges = list(grid.edges(data=True))
    num_branch = len(edges)
    nodes_list = list(grid.nodes)

    # 1. จัดกลุ่มอุปกรณ์เพื่อสร้างตัวแปร CVXPY
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
    SoC = cp.Variable((num_bat, T)) if num_bat > 0 else None

    constraints = []
    slack_idx = next(idx for idx, b in enumerate(grid.buses) if b.type == 'Slack')

    # ====================================================================
    # 🌟 4. SWITCHABLE RELAXATION CORE
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
    # 🌟 5. COMMON DEVICE CONSTRAINTS (Time-Coupling for 24 Hours)
    # ====================================================================
    for t in range(T):
        
        # 5.1 Generator Limits (Active and Reactive Power Bounds)
        if num_gen > 0:
            for k, m in enumerate(machines):
                constraints.extend([
                    Pg[k, t] >= (m.Pmin if m.Pmin != float('-inf') else 0.0) / grid.Sbase,
                    Pg[k, t] <= (m.Pmax if m.Pmax != float('inf') else 9999.0) / grid.Sbase,
                    Qg[k, t] >= (m.Qmin if m.Qmin != float('-inf') else -9999.0) / grid.Sbase,
                    Qg[k, t] <= (m.Qmax if m.Qmax != float('inf') else 9999.0) / grid.Sbase
                ])

        # 5.2 Smart Inverter Limits (Active Curtailment & Apparent Power Cone)
        if num_inv > 0:
            for k, inv in enumerate(inverters):
                constraints.extend([
                    P_inv[k, t] >= 0,
                    P_inv[k, t] <= P_solar_avail_pu[k, t],  # Allow solar curtailment if grid is congested
                    cp.norm(cp.vstack([P_inv[k, t], Q_inv[k, t]])) <= inv.S_max / grid.Sbase # Volt-VAR capacity
                ])

        # 5.3 Battery Limits & Time-Coupling Dynamics (State of Charge)
        if num_bat > 0:
            for k, bat in enumerate(batteries):
                p_max_pu = bat.P_max / grid.Sbase
                e_max_pu = bat.E_max / grid.Sbase
                
                constraints.extend([
                    P_ch[k, t] >= 0, P_ch[k, t] <= p_max_pu,
                    P_dis[k, t] >= 0, P_dis[k, t] <= p_max_pu,
                    SoC[k, t] >= 0.1, SoC[k, t] <= 1.0  # Maintain at least 10% reserve capacity
                ])

                # SoC Inter-temporal constraint (Energy balance across time)
                delta_soc = (P_ch[k, t] * bat.eta - P_dis[k, t] / bat.eta) / e_max_pu
                if t == 0:
                    constraints.append(SoC[k, t] == bat.init_soc + delta_soc)
                else:
                    constraints.append(SoC[k, t] == SoC[k, t-1] + delta_soc)

    # ====================================================================
    # 🌟 6. OBJECTIVE FUNCTION & SOLVE
    # ====================================================================
    cost = 0
    for t in range(T):
        
        # 6.1 Generator Operating Cost (Quadratic formulation)
        if num_gen > 0:
            for k, m in enumerate(machines):
                Pg_mw = Pg[k, t] * grid.Sbase
                cost += m.a + (m.b * Pg_mw) + (m.c * cp.square(Pg_mw))
        
        # 6.2 Renewable Incentives & ESS Degradation
        if num_inv > 0:
            # Reward AI slightly for utilizing solar energy (Negative cost)
            for k, inv in enumerate(inverters): 
                cost -= 0.01 * (P_inv[k, t] * grid.Sbase)
                
        if num_bat > 0:
            # Apply minor degradation cost to prevent unnecessary charge/discharge cycles
            for k, bat in enumerate(batteries): 
                cost += 0.1 * (P_ch[k, t] + P_dis[k, t]) * grid.Sbase

    # Initialize and Solve the optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"\n⏳ Solving 24-Hr Master Plan with MPOPF ({relax.upper()} Relaxation)...")
    
    try:
        prob.solve(solver=getattr(cp, solver_name.upper()), verbose=False)
    except Exception as e:
        print(f"❌ CVXPY Error: {e}")

    # Extract results and populate hardware instances
    if prob.status in ["optimal", "optimal_inaccurate"]:
        print(f"✅ MPOPF ({relax.upper()}) Converged! Master Plan 24h Total Cost: ${prob.value:.2f}")
        
        if num_gen > 0:
            for k, m in enumerate(machines):
                m.P_series = np.array(Pg.value[k, :]) * grid.Sbase
                m.Q_series = np.array(Qg.value[k, :]) * grid.Sbase
        if num_inv > 0:
            for k, inv in enumerate(inverters):
                inv.P_series = np.array(P_inv.value[k, :]) * grid.Sbase
                inv.Q_series = np.array(Q_inv.value[k, :]) * grid.Sbase
        if num_bat > 0:
            for k, bat in enumerate(batteries):
                bat.P_ch_series = np.array(P_ch.value[k, :]) * grid.Sbase
                bat.P_dis_series = np.array(P_dis.value[k, :]) * grid.Sbase
                bat.SoC_series = np.array(SoC.value[k, :])
    else:
        raise Exception(f"❌ MPOPF Infeasible! AI could not find a safe 24h plan (Status: {prob.status})")

def NR(grid, tol=1e-6, max_iter=100):
    print("\n🚀 Starting Newton-Raphson Power Flow...")
    
    Ybus = grid.build_ybus()
    num_bus = len(grid.buses)
    
    iteration = 0
    converged = False
    
    while iteration < max_iter:
        # =======================================================
        # 1. Update Voltage Dependencee
        # =======================================================
        for bus in grid.buses:
            for comp in bus.components:
                if isinstance(comp, AsynchronousMachine):
                    grid.update_motor_slip(comp, bus.V)
                    comp.update_pq_from_slip(bus.V, grid.Sbase)
                elif isinstance(comp, Shunt):
                    comp.update_voltage_dependence(bus.V)

        # =======================================================
        # 2. Classify Dynamic Bus
        # =======================================================
        slack, pv, pq = [], [], []
        for i, bus in enumerate(grid.buses):
            if bus.type == 'Slack': slack.append(i)
            elif bus.type == 'PV': pv.append(i)
            else: pq.append(i)
        non_slack = pv + pq

        # =======================================================
        # 3. Prepare V, theta and Spec
        # =======================================================
        V = np.array([b.V for b in grid.buses], dtype=float)
        theta = np.array([b.theta for b in grid.buses], dtype=float)
        
        P_spec = np.zeros(num_bus)
        Q_spec = np.zeros(num_bus)
        
        for i, bus in enumerate(grid.buses):
            for comp in bus.components:
                c_type = type(comp).__name__
                if c_type == 'SynchronousMachine':
                    P_spec[i] += comp.P / grid.Sbase
                    Q_spec[i] += comp.Q / grid.Sbase
                elif c_type == 'Load':
                    P_spec[i] -= abs(comp.P) / grid.Sbase
                    Q_spec[i] -= abs(comp.Q) / grid.Sbase
                elif c_type == 'Inverter':
                    P_spec[i] += abs(comp.P) / grid.Sbase
                    Q_spec[i] += abs(comp.Q) / grid.Sbase
                elif c_type == 'AsynchronousMachine':
                    P_spec[i] -= abs(comp.P) / grid.Sbase
                    Q_spec[i] -= abs(comp.Q) / grid.Sbase
                elif c_type == 'Shunt':
                    Q_spec[i] += comp.Q / grid.Sbase

        # ================================
        # 4. Calculatee P_calc, Q_calc
        # ================================
        Vc = V * np.exp(1j * theta)
        I = Ybus @ Vc
        S_calc = Vc * np.conj(I)
        
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        # =====================
        # 5. Q-Limit Check
        # =====================
        limit_hit_this_iter = False
        for i in pv:
            bus = grid.buses[i]
            Q_load_pu = sum([abs(c.Q) for c in bus.components if isinstance(c, (Load, AsynchronousMachine))]) / grid.Sbase
            Q_gen_req_pu = Q_calc[i] + Q_load_pu  # พลังงานที่เจเนอเรเตอร์ต้องผลิต

            machines = [c for c in bus.components if isinstance(c, SynchronousMachine)]
            if machines:
                Qmax_total_pu = sum(getattr(m, 'Qmax', 9999.0) for m in machines) / grid.Sbase
                Qmin_total_pu = sum(getattr(m, 'Qmin', -9999.0) for m in machines) / grid.Sbase
                
                if Q_gen_req_pu > Qmax_total_pu:
                    print(f"⚠️ Iteration {iteration}: Bus {bus.name} exceed Qmax ({Q_gen_req_pu*grid.Sbase:.2f} > {Qmax_total_pu*grid.Sbase:.2f}) -> Change to PQ Bus!")
                    bus.type = 'PQ'
                    for m in machines: m.Q = getattr(m, 'Qmax', 9999.0)
                    limit_hit_this_iter = True
                        
                elif Q_gen_req_pu < Qmin_total_pu:
                    print(f"⚠️ Iteration {iteration}: Bus {bus.name} exceed Qmin ({Q_gen_req_pu*grid.Sbase:.2f} < {Qmin_total_pu*grid.Sbase:.2f}) -> Change to PQ Bus!")
                    bus.type = 'PQ'
                    for m in machines: m.Q = getattr(m, 'Qmin', -9999.0)
                    limit_hit_this_iter = True

        if limit_hit_this_iter:
            iteration += 1
            continue

        # =======================================================
        # 6. Find Mismatch and Convergence check
        # =======================================================
        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc
        
        dP_ns = dP[non_slack]
        dQ_pq = dQ[pq]
        mismatch = np.concatenate((dP_ns, dQ_pq))
        
        max_error = np.max(np.abs(mismatch)) if len(mismatch) > 0 else 0
        if max_error < tol:
            converged = True
            break # 🌟 หลุดจากลูป NR ได้!
            
        # =======================================================
        # 7. Jacobian Matrix
        # =======================================================
        diagVc = np.diag(Vc)
        diagI_conj = np.diag(np.conj(I))
        diagVc_conj = np.diag(np.conj(Vc))
        diagV_phase = np.diag(Vc / V)
        diagV_phase_conj = np.diag(np.conj(Vc) / V)
        
        dS_dTheta = 1j * diagVc @ diagI_conj - 1j * diagVc @ np.conj(Ybus) @ diagVc_conj
        dS_dV = diagV_phase @ diagI_conj + diagVc @ np.conj(Ybus) @ diagV_phase_conj
        
        H = dS_dTheta.real
        N = dS_dV.real
        M = dS_dTheta.imag
        L = dS_dV.imag
        
        J11 = H[np.ix_(non_slack, non_slack)]
        J12 = N[np.ix_(non_slack, pq)]
        J21 = M[np.ix_(pq, non_slack)]
        J22 = L[np.ix_(pq, pq)]
        
        J_top = np.hstack((J11, J12)) if J12.size else J11
        J_bottom = np.hstack((J21, J22)) if J12.size else J21
        J = np.vstack((J_top, J_bottom))
        
        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            print("❌ Singular Jacobian Matrix! The grid might be physically collapsed.")
            break
            
        dTheta = dx[:len(non_slack)]
        dV = dx[len(non_slack):]
        
        for idx, i in enumerate(non_slack): grid.buses[i].theta += dTheta[idx]
        for idx, i in enumerate(pq): grid.buses[i].V += dV[idx]

        # =============================
        # 9. OLTC Auto-Tap Check
        # =============================
        tap_changed = False
        for u, v, data in grid.edges(data=True):
            branch = data.get('obj')
            if isinstance(branch, Transformer) and getattr(branch, 'auto_tap', False) and getattr(branch, 'controlled_bus', None):
                target_bus = grid.bus(branch.controlled_bus)
                deadband = 0.015 
                
                if target_bus.V < branch.target_V - deadband and branch.tap_ratio > branch.tap_min:
                    branch.tap_ratio -= branch.tap_step
                    branch.tap_ratio = max(branch.tap_ratio, branch.tap_min)
                    tap_changed = True
                    print(f"🔄 Iteration {iteration}: {branch.name} Step DOWN Tap -> {branch.tap_ratio:.4f}")
                    
                elif target_bus.V > branch.target_V + deadband and branch.tap_ratio < branch.tap_max:
                    branch.tap_ratio += branch.tap_step
                    branch.tap_ratio = min(branch.tap_ratio, branch.tap_max)
                    tap_changed = True
                    print(f"🔄 Iteration {iteration}: {branch.name} Step UP Tap -> {branch.tap_ratio:.4f}")

        # ถ้าสับ Tap ต้องสร้าง Ybus ใหม่
        if tap_changed:
            Ybus = grid.build_ybus()

        iteration += 1
        
    # ===========================
    # 10. Post-Processing 
    # ===========================
    V_final = np.array([b.V for b in grid.buses])
    theta_final = np.array([b.theta for b in grid.buses])
    Vc_final = V_final * np.exp(1j * theta_final)
    S_final = Vc_final * np.conj(Ybus @ Vc_final)
    
    for i, bus in enumerate(grid.buses):
        P_load_total = sum([abs(c.P)/grid.Sbase for c in bus.components if type(c).__name__ in ['Load', 'AsynchronousMachine']])
        Q_load_total = sum([abs(c.Q)/grid.Sbase for c in bus.components if type(c).__name__ in ['Load', 'AsynchronousMachine']])
        P_inv_total = sum([abs(c.P)/grid.Sbase for c in bus.components if type(c).__name__ == 'Inverter'])
        Q_inv_total = sum([abs(c.Q)/grid.Sbase for c in bus.components if type(c).__name__ == 'Inverter'])
        Q_shunt_total = sum([c.Q/grid.Sbase for c in bus.components if type(c).__name__ == 'Shunt'])
        
        if bus.type == 'Slack':
            P_slack_gen = S_final.real[i] + P_load_total - P_inv_total
            Q_slack_gen = S_final.imag[i] + Q_load_total - Q_inv_total - Q_shunt_total
            for comp in bus.components:
                if type(comp).__name__ == 'SynchronousMachine':
                    comp.P = float(P_slack_gen * grid.Sbase)
                    comp.Q = float(Q_slack_gen * grid.Sbase)
                    
        elif bus.type == 'PV':
            Q_pv_gen = S_final.imag[i] + Q_load_total - Q_inv_total - Q_shunt_total
            for comp in bus.components:
                if type(comp).__name__ == 'SynchronousMachine':
                    comp.Q = float(Q_pv_gen * grid.Sbase)

    if converged:
        print(f"✅ Newton-Raphson Converged seamlessly in {iteration+1} iterations!")
    else:
        print(f"⚠️ Newton-Raphson Failed after {max_iter} iterations (Max Error: {max_error:.6f})")
        
def plan(grid, profile_df:DataFrame, path:str = "settings.json", relax:str='SOCP', solver:str = 'SCS', tol:float=1e-6, max_iter:int=100):
    print(f"\n🔮 Starting Modern Grid Simulation (MPOPF) for {len(profile_df)} steps...")
        
    MPOPF(grid, profile_df, relax, solver)
        
    history = []
        
    for t in range(len(profile_df)):
        print(f"\n{'='*15} 🕒 Step {t} {'='*15}")
            
        grid.apply_profile(profile_df.iloc[t].to_dict())
            
        for bus in grid.buses:
            for comp in bus.components:
                c_name = type(comp).__name__
                if c_name == 'SynchronousMachine':
                    comp.P = float(comp.P_series[t])
                    comp.Q = float(comp.Q_series[t])
                elif c_name == 'Inverter':
                    comp.P = float(comp.P_series[t])
                    comp.Q = float(comp.Q_series[t])
                elif c_name == 'Battery':
                    comp.P = float(comp.P_dis_series[t] - comp.P_ch_series[t])  # Net Injection
                    comp.SoC = float(comp.SoC_series[t])
            
        NR(grid, max_iter=max_iter, tol=tol)
            
        total_cost = sum([bus.total_cost() for bus in grid.buses])
            
        step_result = {
            'step': t,
            'system': {'total_cost_per_hr': round(total_cost, 2)},
            'buses': {}, 'components': {}, 'branches': {}
        }
            
        for bus in grid.buses:
            step_result['buses'][bus.name] = {
                'V_pu': round(bus.V, 4), 'theta_rad': round(bus.theta, 4),
                'P_total_MW': round(bus.P * grid.Sbase, 2), 'Q_total_MVAr': round(bus.Q * grid.Sbase, 2)
            }
            for comp in bus.components:
                comp_data = {
                    'bus': bus.name, 'type': type(comp).__name__,
                    'P_MW': round(comp.P * grid.Sbase, 2) if hasattr(comp, 'P') else 0.0,
                    'Q_MVAr': round(comp.Q * grid.Sbase, 2) if hasattr(comp, 'Q') else 0.0
                }
                if type(comp).__name__ == 'Battery':
                    comp_data['SoC'] = round(comp.SoC, 4)
                step_result['components'][comp.name] = comp_data
            
        for u, v, data in grid.edges(data=True):
            branch = data.get('obj')
            if branch:
                from_bus, to_bus = grid.bus(u), grid.bus(v)
                V_i, V_j = from_bus.V * np.exp(1j * from_bus.theta), to_bus.V * np.exp(1j * to_bus.theta)
                if type(branch).__name__ == 'Transformer':
                    t_tap = branch.tap_ratio * np.exp(1j * branch.phase_shift)
                    I_ij = (V_i/t_tap - V_j) * branch.Y
                else:
                    I_ij = (V_i - V_j) * branch.Y
                    
                S_flow_MVA = abs(V_i * np.conj(I_ij)) * grid.Sbase
                branch_data = {
                    'from_bus': u, 'to_bus': v, 'type': type(branch).__name__,
                    'flow_MVA': round(S_flow_MVA, 2)
                }
                if getattr(branch, 'S_max', None):
                    loading_pct = (S_flow_MVA / branch.S_max) * 100
                    branch_data['loading_percent'] = round(loading_pct, 2)
                    branch_data['is_overload'] = loading_pct > 100
                step_result['branches'][branch.name] = branch_data
                    
        history.append(step_result)
    
    grid.loading_status()
    
    result = json.dumps(history, indent=2, cls=NumpyEncoder)
    with open(path, "w") as f:
        f.write(result)
            
    print("\n✅ MPOPF Time-Series Simulation Completed!")
    return result