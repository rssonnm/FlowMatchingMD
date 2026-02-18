import torch
import torch.nn as nn
import torch.nn.functional as F
from .riemannian_flow import so3_exp_map, skew_symmetric

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        # t: [1] or [B]
        if t.dim() == 0: t = t.unsqueeze(0)
        if t.dim() == 1: t = t.unsqueeze(-1)
        return self.mlp(t)

class InteractionAwareEquivariantBlock(nn.Module):
    """
    Block Transformer với cơ chế Attention nhận biết tương tác hóa học.
    """
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Interaction Embedding:
        # Distance (RBF) + Chemical Match (Donor-Acceptor, Hydrophobic-Hydrophobic)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1 + 3, hidden_dim), # distance + 3 interaction types
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim) # Residual update usually
        )

    def forward(self, h_l, x_l, h_r, x_r):
        # h_l: [N_l, D], x_l: [N_l, 3]
        # h_r: [N_r, D], x_r: [N_r, 3]
        
        # Assume last 3 dims of h are: [IsDonor, IsAcceptor, IsHydrophobic]
        # h_l[..., -3:]
        
        # 1. Distance
        dist = torch.cdist(x_l, x_r) # [N_l, N_r]
        
        # 2. Extract Chemical Props (assuming they are at the end of feature vector)
        # We need to make sure node_in_dim covers this. 
        # In data_processing, we added 3 features.
        l_props = h_l[:, -3:] # [N_l, 3]
        r_props = h_r[:, -3:] # [N_r, 3]
        
        # 3. Compute Interaction Matrices [N_l, N_r]
        # H-bond: Donor(L) - Acceptor(R) OR Acceptor(L) - Donor(R)
        # Note: logic is simplify, multipy probabilities
        hbond_matrix = (l_props[:, 0:1] * r_props[:, 1:2].T) + (l_props[:, 1:2] * r_props[:, 0:1].T) 
        
        # Hydrophobic: Hydrophobic(L) - Hydrophobic(R)
        hydro_matrix = l_props[:, 2:3] * r_props[:, 2:3].T
        
        # Electrostatic-like (just using simple mismatch/match logic or raw charge product)
        # For now, let's stick to Hbond + Hydro + Generic
        
        # Stack edge features: [N_l, N_r, 3]
        # Dist, Hbond, Hydro, 1 (bias)
        # Use simple tensor operations
        
        # Edge Feature Vector Construction
        # [N_l, N_r, 1]
        dist_feat = torch.exp(-dist[..., None]**2 / 5.0) # Sharp RBF
        
        # We need to broadcast properly
        # hbond_matrix: [N_l, N_r]
        edge_raw = torch.stack([
            dist_feat.squeeze(-1),
            hbond_matrix,
            hydro_matrix,
            torch.ones_like(dist) # Bias/Base connection
        ], dim=-1) # [N_l, N_r, 4]
        
        # Compute Attention Weights / Message Weights
        # MLP([dist, hbond, hydro, bias]) -> [N_l, N_r, Hidden]
        
        # Since input to MLP is only 4 dims, we adjust Linear layer above.
        # Wait, previous code was Linear(3 + node_dim * 2).
        # We replace it with specialized Interaction MLP.
        
        interaction_embed = self.edge_mlp(edge_raw) # [N_l, N_r, Hidden]
        
        # Mask by distance to save compute? (Soft mask via RBF already done implicitly)
        
        # Aggregate: Sum(Interaction * h_r)
        # h_r: [N_r, D] -> [1, N_r, D]
        # interaction_embed: [N_l, N_r, Hidden] (assuming Hidden=D for elementwise mult)
        # If Hidden != D, we project h_r first.
        # Let's assume hidden_dim == node_dim for simplicity or project h_r
        
        # Simple attention: weights = sigmoid(interaction_embed)
        weights = torch.sigmoid(interaction_embed) # [N_l, N_r, Hidden]
        
        # We need to broadcast h_r to [N_l, N_r, D]
        # If Hidden != D, this logic is tricky.
        # Let's project h_r to Hidden
        h_r_proj = h_r # Simplified, usually strictly defined dimensions
        
        # Weighted Sum
        # out[i] = Sum_j ( weight_ij * h_r_j )
        # [N_l, N_r, D] * [1, N_r, D] (broadcasting)
        
        msg = weights * h_r.unsqueeze(0) 
        h_l_agg = torch.sum(msg, dim=1) # [N_l, D]
        
        # Scale by count?
        h_l_agg = h_l_agg / (torch.sum(weights, dim=1) + 1e-6)
        
        # Update
        h_l_new = self.node_update(torch.cat([h_l, h_l_agg], dim=-1))
        
        return h_l + h_l_new # ResNet

class SE3EquivariantModel(nn.Module):
    """
    Mô hình dự đoán vận tốc (v, w) trên SE(3).
    Dựa trên cấu trúc Equivariant Frame.
    """
    def __init__(self, node_in_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)
        
        self.layers = nn.ModuleList([
            InteractionAwareEquivariantBlock(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Output heads for tangent space SE(3)
        # v (translation velocity): [3]
        # w (angular velocity): [3]
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        self.w_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, h_l, R_t, p_t, h_r, x_r, t, mask=None):
        """
        Args:
            h_l: [N_l, node_in_dim]
            R_t: [3, 3] current rotation of ligand
            p_t: [3] current center of ligand
            h_r: [N_r, node_in_dim]
            x_r: [N_r, 3] receptor atoms (fixed)
            t: [1] time
        """
        # 1. Embeddings
        h_l = self.node_embed(h_l)
        h_r = self.node_embed(h_r)
        
        t_emb = self.time_embed(t).expand(h_l.size(0), -1)
        h_l = h_l + t_emb
        
        # 2. Current global coordinates of ligand atoms
        # x_l_rel are original relative positions (usually stored in h_l or separate)
        # Let's assume positions passed in h_l's first few dims or separate.
        # SOTA models typically use h_l as categorical and x_l as geometric.
        # We need the original shape of ligand.
        # For now, let's assume we have `x_l_local` stored.
        # x_l_t = R_t @ x_l_local + p_t
        
        # TO BE REFINED: How to get x_l_local? 
        # For simplicity, let's assume h_l contains local coords in first 3 dims.
        x_l_local = h_l[:, :3] 
        x_l_t = torch.matmul(x_l_local, R_t.transpose(-1, -2)) + p_t
        
        # 3. Equivariant Processing
        for layer in self.layers:
            h_l = layer(h_l, x_l_t, h_r, x_r)
            
        # 4. Global Aggregation to SE(3) velocity
        # Pooled ligand feature
        h_pool = torch.mean(h_l, dim=0) # [D]
        
        v_raw = self.v_head(h_pool) # [3]
        w_raw = self.w_head(h_pool) # [3]
        
        # Equivariance correction: Project to correct frame
        # Since v, w are predicted in a fixed frame, we must ensure they rotate with R_t.
        # Usually, we predict them in the 'body frame' and then rotate to 'world frame'.
        v_world = torch.matmul(R_t, v_raw)
        w_world = torch.matmul(R_t, w_raw)
        
        return w_world, v_world
