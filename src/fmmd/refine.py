import torch
import torch.nn as nn
from .scoring import PhysicsScoring

class LangevinRefiner:
    """
    Refines ligand pose using Langevin Dynamics on the Physics Energy landscape.
    x_{t+1} = x_t - step_size * grad(E(x)) + sqrt(2 * step_size * T) * noise
    """
    def __init__(self, steps=100, step_size=0.01, temperature=0.1, friction=0.0):
        self.steps = steps
        self.step_size = step_size
        self.temperature = temperature
        self.friction = friction # Not used in simple overdamped
        self.scorer = PhysicsScoring()

    def refine(self, ligand_data, receptor_data):
        """
        Refines the ligand positions in-place (or returns new positions).
        
        Args:
            ligand_data: MolecularData (positions will be optimized)
            receptor_data: MolecularData (fixed)
        
        Returns:
            refined_positions: [N_l, 3]
            energies: List of energies during refinement
        """
        # 1. Prepare Tensors
        # Ligand positions need gradient
        lig_pos = ligand_data.positions.clone().detach().requires_grad_(True)
        rec_pos = receptor_data.positions.detach() # Fixed
        
        l_charges = ligand_data.charges
        r_charges = receptor_data.charges
        l_types = ligand_data.atom_types
        r_types = receptor_data.atom_types
        
        energies = []
        
        optimizer = torch.optim.SGD([lig_pos], lr=self.step_size)
        # Typically Langevin is manual update, but SGD is close to gradient descent part.
        # We will do manual update to add noise.
        
        for i in range(self.steps):
            optimizer.zero_grad()
            
            total_energy, terms = self.scorer.calculate_energy(
                lig_pos, rec_pos, l_charges, r_charges, l_types, r_types, return_tensor=True
            )
            
            total_energy.backward()
            
            # Gradient Clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([lig_pos], max_norm=1.0)
            
            optimizer.step()
            
            # Add Noise (Langevin Dynamics)
            # noise ~ N(0, sqrt(2 * step_size * T))
            sigma_noise = torch.sqrt(torch.tensor(2 * self.step_size * self.temperature, device=lig_pos.device))
            noise = torch.randn_like(lig_pos, device=lig_pos.device) * sigma_noise
            
            with torch.no_grad():
                lig_pos.add_(noise)
                
            energies.append(total_energy.item())
        
        return lig_pos.detach(), energies
