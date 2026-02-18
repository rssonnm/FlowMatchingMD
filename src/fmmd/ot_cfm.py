import torch

class OTConditionalFlowMatcher:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).
    Reference: Tong et al., "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport", 2023.
    
    Target path: x_t = (1 - t) * x0 + t * x1
    Target vector field: u_t(x|x0, x1) = x1 - x0
    """
    def __init__(self, sigma_min=0.0):
        self.sigma_min = sigma_min

    def sample_location_and_target(self, x0, x1):
        """
        Sample x_t and calculate target v_t.
        
        Args:
            x0: [N, D] Source distribution (Noise)
            x1: [N, D] Target distribution (Data)
            
        Returns:
            t: [N] or [1] time
            xt: [N, D] sample at time t
            ut: [N, D] target velocity
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # 1. Sample time t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device).unsqueeze(1) # [N, 1]
        
        # 2. Optimal Transport Path interpolation
        # xt = (1 - (1 - sigma_min) * t) * x0 + t * x1  <-- General formulation
        # For simple OT-CFM where sigma_min=0 (Dirac path):
        # xt = (1 - t) * x0 + t * x1
        
        # We add small sigma_min for stability if needed, usually 0.0 is fine for OT-CFM 
        # but Tong paper suggests small sigma for "Conditional" FM foundation.
        # But pure OT path is straight line.
        
        xt = (1 - t) * x0 + t * x1
        
        # 3. Target Velocity
        # dx/dt = x1 - x0
        ut = x1 - x0
        
        return t.squeeze(1), xt, ut

    def loss_fn(self, model, x1, x0, h, edge_index, edge_attr, mask, **kwargs):
        """
        Compute CFM Loss.
        """
        t, xt, ut = self.sample_location_and_target(x0, x1)
        
        # Model Prediction: v_theta(xt, t)
        # Note: model expects t as [Batch] or [1]
        # In our current model implementation, t is passed as is.
        
        # Since our nodes are in a single graph (Batch=1 essentially), 
        # we might need to handle t being per-node or scalar.
        # Current model.forward accepts t.
        # If t is [N], we need to ensure model handles it.
        # Let's average t for the batch/graph for simplicity if needed, 
        # or assume model broadcasts t.
        
        # For simplicity in this codebase: sample ONE t for the whole graph
        t_scalar = torch.rand(1, device=x1.device)
        xt = (1 - t_scalar) * x0 + t_scalar * x1
        ut = x1 - x0
        
        vt = model(h, xt, edge_index, edge_attr, t_scalar, mask=mask)
        
        # Squared Error
        loss = torch.mean((vt[mask] - ut[mask])**2)
        return loss
