import torch
import numpy as np
from scipy.spatial.distance import cdist

class PhysicsScoring:
    """
    Hàm chấm điểm dựa trên vật lý đơn giản (Physics-based scoring).
    Bao gồm Lennard-Jones (VDW) và Coulomb (Electrostatic).
    """
    def __init__(self, epsilon=0.1, sigma=3.5, dielectric=10.0):
        # Tham số Lennard-Jones mặc định (giả định chung cho C-C)
        self.epsilon = epsilon # kcal/mol
        self.sigma = sigma     # Angstrom
        self.dielectric = dielectric
        
        # Hằng số Coulomb: 332.06 kcal/mol * A / e^2
        self.COULOMB_CONST = 332.06

    def calculate_energy(self, ligand_pos, receptor_pos, ligand_charges, receptor_charges, atom_types_l, atom_types_r, return_tensor=False):
        """
        Tính năng lượng liên kết tổng hợp.
        
        Args:
            ligand_pos: [N_l, 3]
            receptor_pos: [N_r, 3]
            ligand_charges: [N_l]
            receptor_charges: [N_r]
            return_tensor: Boolean, return tensor for backprop?
            
        Returns:
            total_energy, terms (dictionary)
        """
        # Tính khoảng cách đôi một
        # dist: [N_l, N_r]
        dist = torch.cdist(ligand_pos, receptor_pos)
        
        # Tránh chia cho 0 (Clash cực đại)
        dist = torch.clamp(dist, min=0.1)
        
        # 1. Lennard-Jones Potential
        # E_LJ = 4*eps * ((sigma/r)^12 - (sigma/r)^6)
        # Để đơn giản, ta dùng eps, sigma chung. Thực tế cần bảng tham số theo loại nguyên tử.
        
        s_r = self.sigma / dist
        vdw = 4 * self.epsilon * (torch.pow(s_r, 12) - torch.pow(s_r, 6))
        
        # Cutoff cho VDW (thường ~8-10A)
        vdw_mask = dist < 10.0
        e_vdw = torch.sum(vdw * vdw_mask.float())
        
        # 2. Electrostatic Potential
        # E_elec = k * q1 * q2 / (epsilon * r)
        # Tích điện tích [N_l, N_r]
        q_prod = torch.outer(ligand_charges, receptor_charges)
        elec = self.COULOMB_CONST * q_prod / (self.dielectric * dist)
        
        elec_mask = dist < 20.0
        e_elec = torch.sum(elec * elec_mask.float())
        
        total_energy = e_vdw + e_elec
        
        if return_tensor:
            return total_energy, {
                "vdw": e_vdw,
                "electrostatic": e_elec
            }
            
        return total_energy.item(), {
            "vdw": e_vdw.item(),
            "electrostatic": e_elec.item()
        }

class InteractionAnalyzer:
    """
    Phân tích tương tác giữa Ligand và Receptor.
    """
    def analyze(self, ligand, receptor):
        """
        Args:
            ligand: MolecularData
            receptor: MolecularData
            
        Returns:
            list of dict: [{"residue": "TYR", "id": 12, "type": "Hbond", "dist": 2.5}]
        """
        interactions = []
        
        # Chuyển sang numpy để dễ xử lý logic
        l_pos = ligand.positions.detach().cpu().numpy()
        r_pos = receptor.positions.detach().cpu().numpy()
        l_atoms = ligand.atom_types.detach().cpu().numpy()
        r_atoms = receptor.atom_types.detach().cpu().numpy()
        
        # Tính khoảng cách
        dists = cdist(l_pos, r_pos)
        
        # Các ngưỡng
        # Helper to get residue info
        def get_res_info(idx):
            if hasattr(receptor, 'residues') and receptor.residues is not None and idx < len(receptor.residues):
                res_name = receptor.residues[idx]
                res_id = receptor.res_ids[idx] if hasattr(receptor, 'res_ids') and receptor.res_ids is not None else ""
                return f"{res_name}_{res_id}"
            return f"RES_{idx}"

        # Definitions
        # H-Bond
        HBOND_DIST = 3.5
        HBOND_ACCEPTORS = [7, 8, 9, 16] # N, O, F, S
        
        # Salt Bridge
        SALT_BRIDGE_DIST = 4.0
        # Receptor Anions (ASP, GLU) - O
        REC_ANIONS = ['ASP', 'GLU']
        # Receptor Cations (LYS, ARG, HIS) - N
        REC_CATIONS = ['LYS', 'ARG', 'HIS'] 
        
        # Pi-Stacking
        PI_DIST = 5.0 # Centroid distance ideal, but atom-atom proxy
        REC_AROMATIC = ['PHE', 'TYR', 'TRP', 'HIS']
        
        # Loop mainly over close pairs to save time
        # Get all pairs < Max Cutoff (5.0)
        rows, cols = np.where(dists < 5.0)
        
        seen_pairs = set()
        
        for i, j in zip(rows, cols):
            d = dists[i, j]
            l_type = l_atoms[i]
            r_type = r_atoms[j]
            res_str = get_res_info(j)
            res_name = res_str.split('_')[0]
            
            # Get coords for visualization
            r_xyz = r_pos[j].tolist() # [x, y, z]
            
            # 1. Salt Bridge
            # Ligand N (Positive?) <-> Receptor ASP/GLU (O)
            # Ligand O (Negative?) <-> Receptor LYS/ARG/HIS (N)
            # Simplified: N-O interaction with specific residues
            is_salt_bridge = False
            if d < SALT_BRIDGE_DIST:
                # Case A: Ligand (+) -> Receptor (-)
                # Ligand N <-> Receptor O (ASP, GLU)
                if l_type == 7 and r_type == 8 and res_name in REC_ANIONS:
                    is_salt_bridge = True
                # Case B: Ligand (-) -> Receptor (+)
                # Ligand O <-> Receptor N (LYS, ARG, HIS)
                elif l_type == 8 and r_type == 7 and res_name in REC_CATIONS:
                    is_salt_bridge = True
                    
                if is_salt_bridge:
                    pair_key = (j, "Salt Bridge")
                    if pair_key not in seen_pairs:
                        interactions.append({
                            "residue": res_str, 
                            "atom_idx": int(j), 
                            "type": "Salt Bridge", 
                            "distance": round(d, 2),
                            "receptor_coords": r_xyz
                        })
                        seen_pairs.add(pair_key)
                    continue

            # 2. Hydrogen Bond
            if d < HBOND_DIST:
                if l_type in HBOND_ACCEPTORS and r_type in HBOND_ACCEPTORS:
                    pair_key = (j, "Hydrogen Bond")
                    if pair_key not in seen_pairs:
                        interactions.append({
                            "residue": res_str, 
                            "atom_idx": int(j), 
                            "type": "Hydrogen Bond", 
                            "distance": round(d, 2),
                            "receptor_coords": r_xyz
                        })
                        seen_pairs.add(pair_key)
                    continue

            # 3. Pi-Stacking
            # Check if Ligand atom is Aromatic (feature index 2 == 1.0)
            # Check if Receptor residue is Aromatic and atom is Carbon
            if d < PI_DIST:
                is_lig_aromatic = False
                if ligand.features is not None and ligand.features.shape[1] > 2:
                     if ligand.features[i, 2] > 0.5:
                         is_lig_aromatic = True
                
                is_rec_aromatic = (res_name in REC_AROMATIC and r_type == 6)
                
                if is_lig_aromatic and is_rec_aromatic:
                    pair_key = (j, "Pi-Stacking")
                    if pair_key not in seen_pairs:
                        interactions.append({
                            "residue": res_str, 
                            "atom_idx": int(j), 
                            "type": "Pi-Stacking", 
                            "distance": round(d, 2),
                            "receptor_coords": r_xyz
                        })
                        seen_pairs.add(pair_key)
                    continue

            # 4. Hydrophobic
            if d < 4.5:
                if l_type == 6 and r_type == 6:
                    pair_key = (j, "Hydrophobic")
                    if pair_key not in seen_pairs:
                        interactions.append({
                            "residue": res_str, 
                            "atom_idx": int(j), 
                            "type": "Hydrophobic", 
                            "distance": round(d, 2),
                            "receptor_coords": r_xyz
                        })
                        seen_pairs.add(pair_key)
                    continue

        return interactions
