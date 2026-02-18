import torch
import numpy as np

class ODESolver:
    """
    Bộ giải phương trình vi phân thường (ODE Solver).
    Hỗ trợ phương pháp Euler và Runge-Kutta 4 (RK4).
    """
    def __init__(self, model, method='rk4', step_size=0.1):
        self.model = model
        self.method = method
        self.step_size = step_size

from .data_processing import build_graph

class SE3ODESolver:
    """
    ODE Solver chuyên biệt cho đa tạp SE(3).
    Cập nhật R bằng Exp Map và p bằng phép cộng vector.
    """
    def __init__(self, model, step_size=0.1):
        self.model = model
        self.step_size = step_size

    def solve(self, R_init, p_init, h_l, h_r, x_r, t_span=(0, 1), use_repulsion=True):
        t0, t1 = t_span
        num_steps = int((t1 - t0) / self.step_size)
        time_steps = torch.linspace(t0, t1, num_steps + 1, device=p_init.device)
        
        R, p = R_init.clone(), p_init.clone()
        
        from .riemannian_flow import so3_exp_map
        from .scoring import PhysicsScoring
        scorer = PhysicsScoring()
        
        # Local coords for repulsion check
        x_l_local = h_l[:, :3]
        
        for i in range(num_steps):
            t_curr = time_steps[i]
            dt = time_steps[i+1] - t_curr
            
            # 1. Model Velocity
            w, v = self.model(R_t=R, p_t=p, h_l=h_l, h_r=h_r, x_r=x_r, t=t_curr)
            
            # 2. Physics-Informed Gradient (Optional but SOTA)
            # If t is near 1 (docking pose), increase repulsion to prevent clashes
            if use_repulsion and t_curr > 0.8:
                with torch.enable_grad():
                    p_temp = p.detach().requires_grad_(True)
                    # Detach R to check only translational gradient for repulsion
                    # Refinement: We could fix R during this check
                    R_det = R.detach()
                    x_l_t = torch.matmul(x_l_local, R_det.transpose(-1, -2)) + p_temp
                    
                    # Compute repulsive energy (simplified VdW)
                    # We only care about clashes
                    dist = torch.cdist(x_l_t, x_r)
                    clash_energy = torch.sum(torch.clamp(2.5 - dist, min=0.0)**2)
                    
                    if clash_energy > 0:
                        clash_energy.backward()
                        # Move AWAY from clash: v = v - grad
                        v = v - 0.1 * p_temp.grad
            
            # 3. Manifold Update
            R = torch.matmul(R, so3_exp_map(w * dt))
            p = p + v * dt
            
        return R, p

def sample_pose_sota(model, ligand_data, receptor_data, pocket_center, steps=20, device='cpu'):
    """
    inference sử dụng mô hình SOTA SE(3).
    """
    # 1. Khởi tạo trạng thái nhiễu (x0) trên SE(3)
    # Translation nhiễu quanh pocket
    p0 = torch.randn(3, device=device) + pocket_center.to(device)
    # Rotation nhiễu (Identity or Random SO3)
    R0 = torch.eye(3, device=device) 
    
    # 2. Features
    h_l = ligand_data.features.to(device)
    h_r = receptor_data.features.to(device)
    x_r = receptor_data.positions.to(device)
    
    # 3. Solver
    solver = SE3ODESolver(model, step_size=1.0/steps)
    
    R_final, p_final = solver.solve(
        R_init=R0, 
        p_init=p0, 
        h_l=h_l, 
        h_r=h_r, 
        x_r=x_r
    )
    
    # x_final = R_final @ x_local + p_final
    x_l_local = h_l[:, :3] # Giả định như trong model_sota.py
    x_final = torch.matmul(x_l_local, R_final.transpose(-1, -2)) + p_final
    
    return x_final

def sample_pose_cross_attn(model, ligand_data, receptor_data, pocket_center, steps=20, device='cpu'):
    # Prepare Ligand Data
    # 1. Self-Interaction Graph for Ligand
    # We need to build a mini-graph just for ligand atoms if we use self-attention
    # Or just assume full connectivity for small ligands?
    # reuse build_graph but only for ligand?
    # For now, let's just make a simple radius graph or full graph for ligand
    
    pos_l = ligand_data.positions.to(device)
    h_l = ligand_data.features.to(device)
    
    # Simple fully connected ligand graph for self-attention
    # Or distance based
    N_l = pos_l.size(0)
    edge_index_l = torch.combinations(torch.arange(N_l, device=device), r=2).T
    # Add reverse edges
    edge_index_l = torch.cat([edge_index_l, edge_index_l.flip(0)], dim=1)
    
    # Edge attr for ligand (dist)
    # Actually model expects edge_attr. currently just usage RBF or something.
    # For simplicity, we can pass dummy or zeros if model handles it, 
    # but EGNNLayer needs edge_attr of size edge_dim
    # Let's generate RBF for ligand edges
    from .data_processing import gaussian_rbf_1d
    d_l = torch.norm(pos_l[edge_index_l[0]] - pos_l[edge_index_l[1]], dim=1)
    edge_attr_l = gaussian_rbf_1d(d_l, end=10.0, num_rbf=16) # edge_dim=16 matches model
    
    # Prepare Receptor Data
    h_r = receptor_data.features
    x_r = receptor_data.positions.to(device)
    
    if h_r is None:
        # Fallback if no features
        # Create dummy features size [N_r, node_dim]
        # node_dim from model? usually 5.
        rec_feat_dim = h_l.size(1)
        h_r = torch.zeros(x_r.size(0), rec_feat_dim, device=device)
        h_r[:, 0] = 1.0 # arbitrary
    
    h_r = h_r.to(device)

    # Initial x0 for Ligand
    x0_lig = torch.randn_like(pos_l, device=device) + pocket_center.to(device)

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, h_l, h_r, x_r, edge_index_l, edge_attr_l):
            super().__init__()
            self.model = model
            self.h_l = h_l
            self.h_r = h_r
            self.x_r = x_r
            self.edge_index_l = edge_index_l
            self.edge_attr_l = edge_attr_l
            
        def forward(self, x, t):
            # map x (solver state) to x_l
            return self.model(
                h_l=self.h_l, 
                x_l=x, 
                h_r=self.h_r, 
                x_r=self.x_r, 
                t=t, 
                edge_index_l=self.edge_index_l, 
                edge_attr_l=self.edge_attr_l
            )
            
    wrapper = ModelWrapper(model, h_l, h_r, x_r, edge_index_l, edge_attr_l)
    
    # Use Euler for OT-CFM (Straight line path) -> Faster
    solver = ODESolver(wrapper, method='euler', step_size=1.0/steps)
    
    # Clamp velocity in solver?
    # Actually, let's just run. But if untrained, V is huge.
    # We should add a "training_mode" or scale down output?
    # For now, let's clip x_final to be within reasonable bounds of pocket?
    # No, let solver handle it. 
    # But we can add a clamp inside ODESolver step if needed.
    
    x_final, _ = solver.solve(x0_lig) # kwargs are inside wrapper
    
    return x_final

def sample_pose(model, ligand_data, receptor_data, pocket_center, steps=20, device='cpu'):
    # Detect Model Type
    if model.__class__.__name__ == 'CrossAttentionEGNN':
        return sample_pose_cross_attn(model, ligand_data, receptor_data, pocket_center, steps, device)
        
    # Legacy / Standard EGNN sampling
    graph_data = build_graph(ligand_data, receptor_data)
    
    h = graph_data['h'].to(device)
    edge_index = graph_data['edge_index'].to(device)
    edge_attr = graph_data['edge_attr'].to(device)
    mask = graph_data['mask'].to(device) 
    
    x0_lig = torch.randn_like(ligand_data.positions, device=device) + pocket_center.to(device)
    x0_rec = receptor_data.positions.to(device)
    x0_combined = torch.cat([x0_lig, x0_rec], dim=0)
    
    # Wrapper for legacy model to match new ODESolver signature
    class LegacyWrapper(torch.nn.Module):
        def __init__(self, model, h, edge_index, edge_attr, mask):
            super().__init__()
            self.model = model
            self.h = h
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.mask = mask
            
        def forward(self, x, t):
            return self.model(h=self.h, x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, t=t, mask=self.mask)

    wrapper = LegacyWrapper(model, h, edge_index, edge_attr, mask)
    solver = ODESolver(wrapper, method='rk4', step_size=1.0/steps)
    
    x_final, _ = solver.solve(x0_combined)
    
    num_lig = len(ligand_data.positions)
    return x_final[:num_lig]
