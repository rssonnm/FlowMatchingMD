import torch
import torch.nn.functional as F

def skew_symmetric(v):
    """
    Chuyển vector 3D thành ma trận phản đối xứng (skew-symmetric matrix).
    Args:
        v: [..., 3]
    Returns:
        Omega: [..., 3, 3]
    """
    batch_shape = v.shape[:-1]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros(*batch_shape, 3, 3, device=v.device, dtype=v.dtype)
    O[..., 0, 1] = -z
    O[..., 0, 2] = y
    O[..., 1, 0] = z
    O[..., 1, 2] = -x
    O[..., 2, 0] = -y
    O[..., 2, 1] = x
    return O

def so3_exp_map(omega):
    """
    Rodrigues' Rotation Formula: Exp: so(3) -> SO(3)
    Args:
        omega: [..., 3] vector trong không gian tiếp tuyến (axis-angle)
    """
    theta = torch.norm(omega, dim=-1, keepdim=True).unsqueeze(-1) # [..., 1, 1]
    theta_sq = theta**2
    
    # Near zero approximation for stability
    mask = (theta > 1e-6).to(omega.dtype)
    
    # Skew matrix
    K = skew_symmetric(omega)
    K2 = torch.matmul(K, K)
    
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    
    term1 = torch.sin(theta) / (theta + 1e-8)
    term2 = (1 - torch.cos(theta)) / (theta_sq + 1e-8)
    
    R = I + mask * term1 * K + mask * term2 * K2
    # If theta is tiny, Exp(omega) ~ I + K
    R_tiny = I + K
    
    return torch.where(theta > 1e-6, R, R_tiny)

def so3_log_map(R):
    """
    Log: SO(3) -> so(3)
    Args:
        R: [..., 3, 3] ma trận quay
    """
    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1, keepdim=True)
    cos_theta = (tr - 1) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta).unsqueeze(-1) # [..., 1, 1]
    
    # R - Rt
    diff = R - R.transpose(-1, -2)
    
    omega_hat = diff * (theta / (2 * torch.sin(theta) + 1e-8))
    
    omega = torch.stack([
        omega_hat[..., 2, 1],
        omega_hat[..., 0, 2],
        omega_hat[..., 1, 0]
    ], dim=-1)
    
    return omega

class SE3RiemannianFlow:
    """
    Riemannian Flow Matching trên SE(3).
    Dành cho Rigid Docking: Coi ligand là một khối cứng có 6 Degrees of Freedom.
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def sample_geodesic(self, T0, T1, t):
        """
        Lấy mẫu trạng thái T_t trên đường trắc địa nối T0 và T1.
        T = (R, t) trong đó R thuộc SO(3), t thuộc R3.
        
        Args:
            T0: Tuple (R0, p0) - Trạng thái nguồn (Noise/Initial)
            T1: Tuple (R1, p1) - Trạng thái đích (Data/Pose)
            t: Scalar [0, 1]
        """
        R0, p0 = T0
        R1, p1 = T1
        
        # 1. Translation: Euclidean Geodesic
        pt = (1 - t) * p0 + t * p1
        
        # 2. Rotation: SO(3) Geodesic (Slerp-like)
        # Rt = R0 * Exp(t * Log(R0^T * R1))
        delta_R = torch.matmul(R0.transpose(-1, -2), R1)
        omega = so3_log_map(delta_R)
        Rt = torch.matmul(R0, so3_exp_map(t * omega))
        
        # 3. Target Velocity in tangent space
        v_target = p1 - p0
        w_target = omega # Angular velocity in body frame of R0 or world frame? 
                        # Thường là Lie Algebra element.
        
        return (Rt, pt), (w_target, v_target)

    def loss_fn(self, model, T1, T0, h_l, h_r, x_r, **kwargs):
        """
        Riemannian Flow Matching Loss.
        model: SE3EquivariantModel dự đoán (w, v)
        """
        device = h_l.device
        t = torch.rand(1, device=device)
        
        (Rt, pt), (ut_w, ut_v) = self.sample_geodesic(T0, T1, t)
        
        # Predict velocity field
        # Model cần nhận diện được frame (Rt, pt)
        vt_w, vt_v = model(h_l, Rt, pt, h_r, x_r, t, **kwargs)
        
        loss_w = F.mse_loss(vt_w, ut_w)
        loss_v = F.mse_loss(vt_v, ut_v)
        
        return loss_w + loss_v
