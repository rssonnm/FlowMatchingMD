import torch
import torch.nn as nn
import torch.nn.functional as F

class EGNNLayer(nn.Module):
    """
    Lớp E(n)-Equivariant Graph Neural Network (Satorras et al., 2021).
    Cập nhật tọa độ (x) và đặc trưng (h) sao cho bất biến/đồng biến với phép quay/dịch.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        
        # Mạng phi_e: Tính đặc trưng cạnh m_ij
        # Input: h_i, h_j, ||x_i - x_j||^2, edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Mạng phi_x: Tính cập nhật tọa độ (velocity contribution)
        # Input: m_ij
        # Output: scalar weight cho (x_i - x_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False) # Bias False để đảm bảo vector tính chất
        )
        
        # Mạng phi_h: Cập nhật đặc trưng nút
        # Input: h_i, m_i (sum of m_ij)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, h, x, edge_index, edge_attr, node_mask=None):
        """
        Args:
            h: [N, node_dim] - Đặc trưng nút
            x: [N, 3] - Tọa độ nút
            edge_index: [2, E] - Danh sách cạnh (src, dst)
            edge_attr: [E, edge_dim] - Đặc trưng cạnh (vd: RBF khoảng cách ban đầu)
            node_mask: [N, 1] or [N] - Mask for coordinate updates (1=move, 0=fixed)
        """
        row, col = edge_index
        
        # 1. Tính khoảng cách bình phương ||x_i - x_j||^2
        dist_sq = torch.sum((x[row] - x[col])**2, dim=1, keepdim=True)
        
        # 2. Tạo message m_ij
        # Concatenate: [h_i, h_j, dist_sq, edge_attr]
        edge_input = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=1)
        m_ij = self.edge_mlp(edge_input) # [E, hidden_dim]
        
        # 3. Tính cập nhật tọa độ (Equivariant update)
        # v_ij = (x_i - x_j) / (dist + eps) * phi_x(m_ij)
        # Normalize direction
        dist = torch.sqrt(dist_sq + 1e-8)
        w_ij = self.coord_mlp(m_ij)
        
        # Clamp magnitude to prevent explosion
        w_ij = torch.tanh(w_ij) * 1.0
        
        trans = (x[row] - x[col]) / (dist + 1e-8) * w_ij
        
        # Tổng hợp cập nhật coord cho mỗi nút i: sum_j (v_ij)
        # Lưu ý: Cần scatter_add hoặc loop. Ở đây dùng thủ công cho rõ nghĩa.
        # Trong PyTorch Geometric dùng scatter, ở đây ta code thuần PyTorch cho dễ hiểu.
        # x_new = x + sum(trans)
        
        # Aggregation coord
        agg_trans = torch.zeros_like(x)
        agg_trans.index_add_(0, row, trans) # Cộng dồn vào nút nguồn (hoặc đích tùy định nghĩa)
        # Ở đây ta update node 'row' dựa trên neighbors 'col'.
        
        # Apply mask if provided: Only update coordinates for masked nodes (Ligand)
        if node_mask is not None:
             if node_mask.dim() == 1: node_mask = node_mask.unsqueeze(1)
             agg_trans = agg_trans * node_mask
        
        x_new = x + agg_trans
        
        # 4. Cập nhật đặc trưng nút (Invariant update)
        # m_i = sum_j (m_ij)
        agg_msg = torch.zeros(h.size(0), m_ij.size(1), device=h.device)
        agg_msg.index_add_(0, row, m_ij)
        
        node_input = torch.cat([h, agg_msg], dim=1)
        h_new = h + self.node_mlp(node_input) # Residual connection
        
        return h_new, x_new

class FlowMatchingModel(nn.Module):
    """
    Mô hình chính dự đoán trường vector vận tốc v(x, t).
    Sử dụng chuỗi các lớp EGNN.
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=4):
        super().__init__()
        
        # Embedding thời gian t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Embedding input features
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim) # Nếu edge_attr sizes khác nhau
        
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Lớp cuối cùng để ra vận tốc
        # Lưu ý: EGNN update x trực tiếp, nên ta có thể lấy tổng update x làm vận tốc
        pass 

    def forward(self, h, x, edge_index, edge_attr, t, mask=None):
        """
        Dự đoán vận tốc v_t tại vị trí x và thời gian t.
        
        Args:
            h: [N, node_in_dim]
            x: [N, 3]
            t: [Batch_size] hoặc [1]
            mask: [N] - Mask 1 for moving nodes (ligand), 0 for fixed (receptor)
        """
        # 1. Time embedding
        if t.dim() == 0: t = t.unsqueeze(0)
        t_emb = self.time_mlp(t.view(-1, 1)) # [1, hidden] or [B, hidden]
        
        # Broadcast time to nodes (giả sử 1 graph/batch hoặc handle batch index)
        # Simplification: t_emb cộng vào h
        
        h_feat = self.node_embed(h) + t_emb
        edge_feat = self.edge_embed(edge_attr)
        
        x_in = x.clone()
        
        # Qua các lớp EGNN
        for layer in self.layers:
            h_feat, x_new = layer(h_feat, x, edge_index, edge_feat, node_mask=mask)
            x = x_new
            
        # Vận tốc là sự thay đổi tọa độ
        # v = x_out - x_in
        # Tuy nhiên, EGNN layer đã cộng dồn x. 
        # Để chính xác, model nên output v trực tiếp, nhưng EGNN formulation là update x.
        # Ta coi output phi_x là v.
        
        v = x - x_in
        if mask is not None:
            if mask.dim() == 1: mask = mask.unsqueeze(1)
            v = v * mask # Force velocity of receptor to 0
            
        return v

def flow_matching_loss(model, x1, x0, t, h, edge_index, edge_attr, mask_ligand):
    """
    Tính hàm loss cho Flow Matching.
    
    Args:
        x1: [N, 3] - Tọa độ ligand thật (đích)
        x0: [N, 3] - Tọa độ ligand nhiễu (nguồn, Gaussian)
        t: [1] - Thời gian sample
        mask_ligand: [N] - Boolean mask, True nếu là nguyên tử Ligand
        
    Returns:
        loss: Scalar
    """
    # 1. Nội suy tọa độ (Optimal Transport path)
    # x_t = (1 - t) * x0 + t * x1
    # Chỉ nội suy cho ligand, receptor giữ nguyên (coi như x1_receptor = x0_receptor)
    # Nhưng x1 receptor là cố định. x0 receptor là cố định.
    # Vậy receptor không thay đổi: x_t_rec = x1_rec
    
    # Thực tế: x0_ligand ~ Normal, x0_receptor = x1_receptor
    
    # Tính x_t cho toàn bộ (đã xử lý logic x0=x1 cho receptor bên ngoài hoặc ở đây)
    x_t = (1 - t) * x0 + t * x1
    
    # 2. Tính vector đích (Target Velocity)
    # u_t = x1 - x0
    target_v = x1 - x0 
    
    # 3. Model Prediction
    # Input x_t vào model
    pred_v = model(h, x_t, edge_index, edge_attr, t)
    
    # 4. Tính Loss (MSE) chỉ trên các nguyên tử Ligand
    # Receptor không di chuyển nên v_target = 0, pred cũng nên = 0
    # Nhưng ta chỉ quan tâm loss của ligand
    
    loss = torch.sum((pred_v[mask_ligand] - target_v[mask_ligand])**2) / torch.sum(mask_ligand)
    return loss

class CrossAttentionLayer(nn.Module):
    """
    EGNN-based Cross Attention: Ligand nodes attend to Receptor nodes.
    Updates Ligand (h_l, x_l) based on Receptor (h_r, x_r).
    Receptor remains fixed (features and coords).
    """
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        
        # Edge MLP for cross-interaction (Ligand-Receptor)
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, hidden_dim), # +1 for dist_sq
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Coordinate update (Ligand moves relative to Receptor)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Node update (Ligand features update)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, h_l, x_l, h_r, x_r, mask=None):
        """
        h_l: [N_l, D], x_l: [N_l, 3]
        h_r: [N_r, D], x_r: [N_r, 3]
        """
        # Compute pairwise distances (All-to-All for now, or within cutoff)
        # Using cdist for simplicity in this implementation
        # For large proteins, one should use radius_graph/k-nn
        
        # dist: [N_l, N_r]
        dist_sq = torch.cdist(x_l, x_r).pow(2) 
        dist = torch.sqrt(dist_sq + 1e-8)
        
        # Create edge features for all pairs
        # Needs expansion: h_l [N_l, 1, D], h_r [1, N_r, D]
        N_l = h_l.size(0)
        N_r = h_r.size(0)
        
        h_l_exp = h_l.unsqueeze(1).expand(-1, N_r, -1)
        h_r_exp = h_r.unsqueeze(0).expand(N_l, -1, -1)
        dist_sq_exp = dist_sq.unsqueeze(-1)
        
        # Input to Edge MLP: [N_l, N_r, 2*D + 1]
        edge_input = torch.cat([h_l_exp, h_r_exp, dist_sq_exp], dim=-1)
        m_ij = self.edge_mlp(edge_input) # [N_l, N_r, H]
        
        # Attention Mask / Cutoff (Optional but recommended for efficiency/physics)
        # Only attend to closer nodes? For now, dense attention weighted by MLP
        
        # Coordinate Update
        # x_i_new = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        # diff: [N_l, N_r, 3]
        diff_vec = x_l.unsqueeze(1) - x_r.unsqueeze(0) 
        
        w_ij = self.coord_mlp(m_ij) # [N_l, N_r, 1]
        w_ij = torch.tanh(w_ij) * 1.0 # Stability
        
        trans = diff_vec / (dist.unsqueeze(-1) + 1e-8) * w_ij
        
        # Aggregation: Sum over j (Receptor nodes)
        agg_trans = torch.sum(trans, dim=1) # [N_l, 3]
        
        x_l_new = x_l + agg_trans
        
        # Feature Update
        agg_msg = torch.sum(m_ij, dim=1) # [N_l, H]
        node_input = torch.cat([h_l, agg_msg], dim=1)
        h_l_new = h_l + self.node_mlp(node_input)
        
        return h_l_new, x_l_new

class CrossAttentionEGNN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, num_layers=3):
        super().__init__()
        
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 1. Self-Attention (Ligand-Ligand) - Using standard EGNNLayer
            # edge_dim defaults to 16 (RBF)
            self_layer = EGNNLayer(hidden_dim, 16, hidden_dim)
            # 2. Cross-Attention (Ligand-Receptor)
            cross_layer = CrossAttentionLayer(hidden_dim, hidden_dim)
            self.layers.append(nn.ModuleList([self_layer, cross_layer]))
            
    def forward(self, h_l, x_l, h_r, x_r, t, edge_index_l, edge_attr_l):
        # Embeddings
        t_emb = self.time_mlp(t.view(-1, 1))
        h_l = self.node_embed(h_l) + t_emb
        h_r = self.node_embed(h_r) # Receptor doesn't necessarily need time, but consistency
        
        x_l_in = x_l.clone()
        
        for self_layer, cross_layer in self.layers:
            # 1. Self-Interaction (Ligand Internal)
            # edge_index_l, edge_attr_l are for ligand-ligand graph
            h_l, x_l = self_layer(h_l, x_l, edge_index_l, edge_attr_l)
            
            # 2. Cross-Interaction (Ligand-Receptor)
            h_l, x_l = cross_layer(h_l, x_l, h_r, x_r)
            
        v = x_l - x_l_in
        return v
