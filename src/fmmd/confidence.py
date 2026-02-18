import torch
import torch.nn as nn
from .model_sota import InteractionAwareEquivariantBlock

class ConfidenceModel(nn.Module):
    """
    Dự đoán độ tin cậy (Confidence Score/pLDDT) cho một pose đã gắn (ligand pose).
    Dựa trên cấu trúc Invariant Transformer.
    """
    def __init__(self, node_in_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            InteractionAwareEquivariantBlock(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Score in [0, 1]
        )

    def forward(self, h_l, x_l, h_r, x_r):
        """
        Args:
            h_l: [N_l, D] Ligand features
            x_l: [N_l, 3] Ligand positions (resulted from docking)
            h_r: [N_r, D] Receptor features
            x_r: [N_r, 3] Receptor positions
        """
        h_l = self.node_embed(h_l)
        h_r = self.node_embed(h_r)
        
        for layer in self.layers:
            h_l = layer(h_l, x_l, h_r, x_r)
            
        # Global aggregation
        h_pool = torch.mean(h_l, dim=0)
        
        confidence = self.score_head(h_pool)
        return confidence

def compute_confidence_loss(pred_conf, x_pred, x_true, threshold=2.0):
    """
    Tính loss cho confidence model dựa trên RMSD thật.
    pred_conf: Dự đoán của model [0, 1]
    x_pred: Pose dự đoán
    x_true: Pose thật (Ground Truth)
    """
    rmsd = torch.sqrt(torch.mean(torch.sum((x_pred - x_true)**2, dim=-1)))
    
    # Target: 1.0 nếu RMSD < threshold (vd 2A), 0.0 nếu lớn hơn.
    # Hoặc dạn hàm logistic mượt:
    target_conf = torch.exp(-rmsd / threshold)
    
    loss = F.binary_cross_entropy(pred_conf, target_conf.detach())
    return loss
