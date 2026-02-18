import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix

class MolecularData:
    """
    Lớp đại diện cho dữ liệu phân tử sau khi xử lý.
    Chứa thông tin về tọa độ, loại nguyên tử, và các đặc trưng khác.
    """
    def __init__(self, atom_types, positions, charges=None, features=None, residues=None, res_ids=None):
        self.atom_types = atom_types  # Tensor: [N, num_atom_types] hoặc [N]
        self.positions = positions    # Tensor: [N, 3]
        self.charges = charges        # Tensor: [N]
        self.features = features      # Tensor: [N, num_features]
        self.residues = residues      # List of str: Residue names ['TYR', 'ALA', ...]
        self.res_ids = res_ids        # List of int: Residue IDs [101, 102, ...]

def parse_ligand_pdb(pdb_file):
    """
    Đọc file PDB của ligand sử dụng RDKit.
    
    Args:
        pdb_file (str): Đường dẫn đến file .pdb
    
    Returns:
        MolecularData: Đối tượng chứa thông tin ligand
    """
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Không thể đọc file PDB: {pdb_file}")

    # Lấy tọa độ
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    
    # Lấy thông tin nguyên tử
    atom_types = []
    charges = []
    
    # Bảng tuần hoàn đơn giản cho one-hot encoding (C, N, O, S, F, P, Cl, Br, I, Other)
    # Đây là ví dụ, có thể mở rộng sau.
    # PART 2: Feature construction sẽ được mở rộng ở đây sau này.
    
    for atom in mol.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        # RDKit có thể tính Gasteiger charges nếu cần, hoặc lấy từ file nếu có
        # Ở đây ta tạm thời để 0 hoặc tính sau
        charges.append(0.0) 

    return MolecularData(
        atom_types=torch.tensor(atom_types, dtype=torch.long),
        positions=torch.tensor(positions, dtype=torch.float32),
        charges=torch.tensor(charges, dtype=torch.float32)
    )

def parse_receptor(file_path):
    """
    Parses receptor file (PDB or PDBQT).
    """
    if file_path.endswith('.pdb'):
        return parse_receptor_pdb_rdkit(file_path)
    else:
        return parse_receptor_pdbqt(file_path)

def parse_receptor_pdb_rdkit(pdb_file):
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read receptor PDB: {pdb_file}")
        
    conf = mol.GetConformer()
    positions = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    
    atom_types = []
    residues = []
    res_ids = []
    
    for atom in mol.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        info = atom.GetPDBResidueInfo()
        if info:
            residues.append(info.GetResidueName().strip())
            res_ids.append(info.GetResidueNumber())
        else:
            residues.append("UNK")
            res_ids.append(0)

    # Simple charge 0 for PDB if not computed (Gasteiger could be computed)
    charges = torch.zeros(len(atom_types))
    
    return MolecularData(
        atom_types=torch.tensor(atom_types, dtype=torch.long),
        positions=positions, # [N, 3]
        charges=charges,
        residues=residues,
        res_ids=res_ids
    )

def parse_receptor_pdbqt(pdbqt_file):
    """
    Đọc file PDBQT của receptor.
    File PDBQT là định dạng của AutoDock, chứa thêm thông tin về điện tích (cột cuối).
    
    Args:
        pdbqt_file (str): Đường dẫn file .pdbqt
        
    Returns:
        MolecularData: Đối tượng chứa thông tin receptor
    """
    positions = []
    atom_types = []  # Lưu dưới dạng atomic number
    charges = []
    
    # Mapping đơn giản từ ký hiệu nguyên tử sang số hiệu nguyên tử
    element_map = {
        'C': 6, 'N': 7, 'O': 8, 'S': 16, 'H': 1, 'F': 9, 
        'P': 15, 'CL': 17, 'BR': 35, 'I': 53, 'MG': 12, 'ZN': 30, 'FE': 26, 'CA': 20
    }

    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # PDB format fixed width parsing
                # x: 30-38, y: 38-46, z: 46-54
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    # Charge thường ở cột cuối cùng trong PDBQT (khoảng 70-76)
                    # Một số file có thể khác, nhưng chuẩn AutoDock là cột sau occupancy/tempFactor
                    # Tuy nhiên PDBQT chuẩn: charge ở 70-76, atom type ở 77-79
                    charge = float(line[70:76]) if len(line) > 76 else 0.0
                    
                    # Atom type
                    element_sym = line[77:79].strip().upper()
                    # Nếu không có ở cuối, thử lấy từ tên nguyên tử (cột 12-14)
                    if not element_sym:
                         element_sym = line[12:14].strip().upper()
                         # Xử lý trường hợp tên như "CA" (Calcium) vs "CA" (C-Alpha)
                         # PDBQT thường rõ ràng hơn. Ta dùng logic đơn giản trước.
                         if element_sym[:1].isalpha() and not element_sym[:2].isalpha():
                             element_sym = element_sym[:1]
                    
                    atomic_num = element_map.get(element_sym, 6) # Mặc định là C nếu không tìm thấy (cần cẩn thận chỗ này)

                    positions.append([x, y, z])
                    atom_types.append(atomic_num)
                    charges.append(charge)
                    
                except ValueError:
                    continue
    
    # Ensure 2D tensor even if empty
    if len(positions) == 0:
        pos_tensor = torch.zeros((0, 3), dtype=torch.float32)
        atom_tensor = torch.zeros((0), dtype=torch.long)
        charge_tensor = torch.zeros((0), dtype=torch.float32)
    else:
        pos_tensor = torch.tensor(positions, dtype=torch.float32)
        atom_tensor = torch.tensor(atom_types, dtype=torch.long)
        charge_tensor = torch.tensor(charges, dtype=torch.float32)

    return MolecularData(
        atom_types=atom_tensor,
        positions=pos_tensor,
        charges=charge_tensor
    )

def get_pocket_center(ligand_data):
    """
    Xác định tâm của túi gắn dựa trên vị trí của ligand (nếu có).
    
    Args:
        ligand_data (MolecularData): Dữ liệu ligand tham chiếu.
        
    Returns:
        torch.Tensor: Tọa độ tâm (x, y, z)
    """
    return torch.mean(ligand_data.positions, dim=0)

def extract_pocket_atoms(receptor_data, center, cutoff=10.0):
    """
    Lấy các nguyên tử receptor nằm trong bán kính cutoff quanh tâm túi gắn.
    Mục đích: Giảm kích thước bài toán, chỉ tập trung vào vùng binding.
    
    Args:
        receptor_data (MolecularData): Dữ liệu toàn bộ receptor.
        center (torch.Tensor): Tâm túi gắn [3].
        cutoff (float): Bán kính cắt (Angstrom).
        
    Returns:
        MolecularData: Dữ liệu receptor đã cắt gọn.
    """
    # Tính khoảng cách từ mọi nguyên tử receptor đến tâm
    dists = torch.norm(receptor_data.positions - center, dim=1)
    mask = dists <= cutoff
    
    # Handle lists (residues, res_ids) if they exist
    residues_masked = None
    res_ids_masked = None
    if receptor_data.residues is not None:
        residues_masked = [receptor_data.residues[i] for i in range(len(mask)) if mask[i]]
    if receptor_data.res_ids is not None:
        res_ids_masked = [receptor_data.res_ids[i] for i in range(len(mask)) if mask[i]]

    return MolecularData(
        atom_types=receptor_data.atom_types[mask],
        positions=receptor_data.positions[mask],
        charges=receptor_data.charges[mask],
        residues=residues_masked,
        res_ids=res_ids_masked
    )

def build_graph(ligand, receptor, cutoff=10.0, num_rbf=16):
    """
    Xây dựng biểu diễn đồ thị cho mô hình GNN.
    Gồm các cạnh nội bộ ligand, nội bộ receptor (nếu cần), và tương tác ligand-receptor.
    
    Args:
        ligand (MolecularData): Dữ liệu ligand.
        receptor (MolecularData): Dữ liệu receptor (đã cắt pocket nếu cần).
        cutoff (float): Khoảng cách cắt để tạo cạnh.
        
    Returns:
        dict: Chứa các tensor cần thiết cho model.
    """
    # 1. Concatenate Nodes
    # Ligand nodes first, then Receptor nodes
    # Mask: 1 for Ligand (moveable), 0 for Receptor (fixed)
    
    num_lig = len(ligand.positions)
    num_rec = len(receptor.positions)
    
    # Positions
    pos = torch.cat([ligand.positions, receptor.positions], dim=0) # [N+M, 3]
    
    # Features (Atom types / attributes)
    # Ensure dimensions match. Ligand has full features?
    # If receptor features are simpler, PAD them or compute same features.
    # For now, let's assume we use 'atom_types' and 'charges' embedding or similar.
    # Or rely on simplified features for standard EGNN.
    
    # Let's use simple features: [AtomType(OneHot), Charge]
    # Or reuse the full features if available. 
    # Ligand has 5 dims in current code. Receptor currently only parsed atom_types/charges.
    
    # Pad receptor features to match ligand feature dim if needed
    feat_dim = ligand.features.size(1) if ligand.features is not None else 1
    
    # Construct Receptor Features (Simple fallback)
    # 1. Atom Type (Mapped later or raw)
    # 2. Charge
    # 3..5. 0.0
    
    if receptor.features is None:
        # Create dummy features for receptor [AtomNum, 0, 0, Charge, 0]
        # Match get_atom_features logic partially
        rec_feat = torch.zeros(num_rec, feat_dim, device=ligand.positions.device)
        # Assuming feat[0] is atom mapping index, feat[3] is charge
        # Using raw atomic num for now in slot 0 if not mapped
        rec_feat[:, 0] = receptor.atom_types.float() 
        rec_feat[:, 3] = receptor.charges
        # Note: Ideally we run `get_atom_features` on Receptor RDKit mol too.
    else:
        rec_feat = receptor.features
        
    h = torch.cat([ligand.features, rec_feat], dim=0) # [N+M, F]
    
    # Create Mask
    # 1.0 for Ligand, 0.0 for Receptor
    mask = torch.cat([torch.ones(num_lig), torch.zeros(num_rec)], dim=0).bool()
    
    # 2. Build Edges (Radius Graph)
    # Calculate pairwise distance
    dist = torch.cdist(pos, pos)
    
    # Get indices where dist < cutoff (and not self-loop)
    src, dst = torch.where((dist < cutoff) & (dist > 1e-4))
    
    edge_index = torch.stack([src, dst], dim=0)
    
    # 3. Edge Attributes (RBF of distances)
    # Re-calculate dists for edges only to save memory if dense
    d_edges = dist[src, dst]
    edge_attr = gaussian_rbf_1d(d_edges, start=0.0, end=cutoff, num_rbf=num_rbf)
    
    return {
        'x': pos,
        'h': h,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'mask': mask,
        'batch': torch.zeros(len(pos), dtype=torch.long) # Single batch
    }

def gaussian_rbf_1d(inputs, start=0.0, end=10.0, num_rbf=16):
    centers = torch.linspace(start, end, num_rbf, device=inputs.device)
    widths = torch.tensor((end / num_rbf), device=inputs.device) # Sigma
    # inputs: [E], centers: [R]
    # [E, 1] - [1, R] -> [E, R]
    return torch.exp(-((inputs.unsqueeze(-1) - centers) ** 2) / (2 * widths**2))

# --- PART 2: Feature Construction ---

def get_atom_features(mol):
    """
    Trích xuất đặc trưng nguyên tử từ RDKit Mol.
    Features:
    - Atomic number (one-hot or integer)
    - Hybridization (one-hot)
    - Aromaticity (bool)
    - Partial charge (float)
    - [NEW] Hydrogen Bond Donor (bool)
    - [NEW] Hydrogen Bond Acceptor (bool)
    - [NEW] Hydrophobic (bool)
    """
    # 1. Chemical Features Factory
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    import os
    
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = featFactory.GetFeaturesForMol(mol)
    
    # Create masks
    num_atoms = mol.GetNumAtoms()
    donor_mask = np.zeros(num_atoms, dtype=bool)
    acceptor_mask = np.zeros(num_atoms, dtype=bool)
    hydrophobic_mask = np.zeros(num_atoms, dtype=bool)
    
    for f in feats:
        family = f.GetFamily()
        indices = f.GetAtomIds()
        for idx in indices:
            if family == 'Donor': donor_mask[idx] = True
            elif family == 'Acceptor': acceptor_mask[idx] = True
            elif family == 'Hydrophobe': hydrophobic_mask[idx] = True
            
    # Define mappings
    ATOM_FAMILIES = [6, 7, 8, 16, 9, 15, 17, 35, 53] # C, N, O, S, F, P, Cl, Br, I
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    
    features = []
    
    # Compute Gasteiger charges if not present
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass # Fallback or ignore if failed

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_feats = []
        
        # 1. Element (One-hotish)
        atomic_num = atom.GetAtomicNum()
        atom_feats.append(ATOM_FAMILIES.index(atomic_num) if atomic_num in ATOM_FAMILIES else len(ATOM_FAMILIES))
        
        # 2. Hybridization (One-hot encoded manually here or index)
        hyb = atom.GetHybridization()
        atom_feats.append(HYBRIDIZATIONS.index(hyb) if hyb in HYBRIDIZATIONS else len(HYBRIDIZATIONS))
        
        # 3. Aromaticity
        atom_feats.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # 4. Partial Charge
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
        except:
            charge = 0.0
        atom_feats.append(charge)
        
        # [NEW] Pharmacophore Props
        atom_feats.append(1.0 if donor_mask[idx] else 0.0)
        atom_feats.append(1.0 if acceptor_mask[idx] else 0.0)
        atom_feats.append(1.0 if hydrophobic_mask[idx] else 0.0)
        
        features.append(atom_feats)

    return torch.tensor(features, dtype=torch.float32)

def gaussian_rbf(inputs, offsets, widths):
    """
    Mã hóa khoảng cách bằng hàm cơ sở xuyên tâm (RBF).
    exp(- (x - \mu)^2 / \sigma^2)
    """
    return torch.exp(-((inputs.unsqueeze(-1) - offsets) ** 2) / widths)

def get_edge_features(pos_src, pos_dst, num_rbf=16, cutoff=10.0):
    """
    Tạo đặc trưng cạnh dựa trên khoảng cách.
    
    Args:
        pos_src: [N, 3]
        pos_dst: [M, 3]
        
    Returns:
        torch.Tensor: [N, M, num_rbf] (hoặc dạng sparse edge_attr)
    """
    # Tính khoảng cách Euclidean
    dist = torch.cdist(pos_src, pos_dst) # [N, M]
    
    # RBF encoding
    # Tạo các tâm (centers) từ 0 đến cutoff
    centers = torch.linspace(0, cutoff, num_rbf)
    widths = torch.tensor((cutoff / num_rbf) ** 2)
    
    # [N, M, 1] - [num_rbf] -> [N, M, num_rbf]
    rbf = gaussian_rbf(dist, centers, widths)
    
    return rbf

# Cập nhật hàm parse_ligand_pdb để dùng atom features
# Cập nhật hàm parse_ligand_pdb để dùng atom features
def parse_ligand_with_features(file_path):
    if file_path.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(file_path, removeHs=False)
        mol = suppl[0] if len(suppl) > 0 else None
    else:
        # Assume PDB
        mol = Chem.MolFromPDBFile(file_path, removeHs=False)
        
    if mol is None:
        raise ValueError(f"Không thể đọc file: {file_path}")

    conf = mol.GetConformer()
    positions = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    features = get_atom_features(mol)
    
    return MolecularData(
        atom_types=torch.tensor(atom_types, dtype=torch.long),
        positions=positions,
        charges=features[:, 3], # Lấy cột charge từ features
        features=features
    )

