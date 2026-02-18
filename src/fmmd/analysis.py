
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix

class InteractionAnalyzer:
    """
    Phân tích chi tiết các tương tác giữa Ligand và Receptor.
    Hỗ trợ: H-bonds, Salt Bridge, Hydrophobic, Pi-Stacking, Metal, Halogen, Clashes.
    """
    def __init__(self, receptor_mol, ligand_mol):
        self.receptor = receptor_mol
        self.ligand = ligand_mol
        
        # Pre-calculate extensive properties
        self._prep_molecules()
        
    def _prep_molecules(self):
        # 1. Get coords
        self.r_conf = self.receptor.GetConformer()
        self.l_conf = self.ligand.GetConformer()
        self.r_pos = self.r_conf.GetPositions()
        self.l_pos = self.l_conf.GetPositions()
        
        # 2. Get Features (Donors, Acceptors, Aromatics, Charges, Hydrophobes)
        self.r_feats = self._get_features(self.receptor)
        self.l_feats = self._get_features(self.ligand)
        
    def _get_features(self, mol):
        from rdkit import RDConfig, Chem
        from rdkit.Chem import ChemicalFeatures
        import os
        
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        feats = featFactory.GetFeaturesForMol(mol)
        
        feature_dict = {
            'Donor': [], 'Acceptor': [], 'Hydrophobe': [], 
            'Aromatic': [], 'PosIonizable': [], 'NegIonizable': [], 
            'Halogen': [], 'Metal': []
        }
        
        for f in feats:
            family = f.GetFamily()
            # Normalize family names if needed
            if family == 'LumpedHydrophobe': family = 'Hydrophobe'
            if family == 'ZnBinder': family = 'Metal'
            if family == 'Aromatics': family = 'Aromatic'
            
            if family in feature_dict:
                # Store (atom_indices, center_pos)
                indices = list(f.GetAtomIds())
                # Calculate geometric center of feature
                pos = np.zeros(3)
                conf = mol.GetConformer()
                for idx in indices:
                    pos += np.array(conf.GetAtomPosition(idx))
                pos /= len(indices)
                
                feature_dict[family].append({
                    'indices': indices,
                    'pos': pos,
                    'type': family
                })

        # Manually add Metals (Zn, Mg, Fe, Ca) as generic 'Metal' features if not caught
        # And Halogens
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            sym = atom.GetSymbol()
            if sym in ['Zn', 'Mg', 'Fe', 'Ca', 'Mn', 'Co', 'Ni', 'Cu']:
                pos = np.array(mol.GetConformer().GetAtomPosition(idx))
                feature_dict['Metal'].append({'indices': [idx], 'pos': pos, 'type': 'Metal'})
            if sym in ['Cl', 'Br', 'I', 'F']:
                 pos = np.array(mol.GetConformer().GetAtomPosition(idx))
                 feature_dict['Halogen'].append({'indices': [idx], 'pos': pos, 'type': 'Halogen'})
        
        return feature_dict

    def analyze(self):
        interactions = []
        
        # 1. Hydrogen Bonds
        # Ligand Donor -> Receptor Acceptor
        interactions.extend(self._check_hbonds(self.l_feats['Donor'], self.r_feats['Acceptor'], "Ligand->Receptor"))
        # Receptor Donor -> Ligand Acceptor
        interactions.extend(self._check_hbonds(self.r_feats['Donor'], self.l_feats['Acceptor'], "Receptor->Ligand"))
        
        # 2. Salt Bridges (Cation <-> Anion)
        # Ligand(+) -> Receptor(-)
        interactions.extend(self._check_salt_bridges(self.l_feats['PosIonizable'], self.r_feats['NegIonizable'])) 
        # Ligand(-) -> Receptor(+)
        interactions.extend(self._check_salt_bridges(self.l_feats['NegIonizable'], self.r_feats['PosIonizable']))
        
        # 3. Hydrophobic
        interactions.extend(self._check_hydrophobic(self.l_feats['Hydrophobe'], self.r_feats['Hydrophobe']))
        
        # 4. Pi-Stacking
        interactions.extend(self._check_pi_stacking(self.l_feats['Aromatic'], self.r_feats['Aromatic']))
        
        # 5. Pi-Cation (To do: implement if needed, usually less critical than Salt Bridge)
        
        # 6. Metal Coordination
        interactions.extend(self._check_metal(self.l_feats, self.r_feats['Metal']))
        
        # 7. Steric Clashes
        interactions.extend(self._check_clashes())

        return interactions

    # ... (Keep existing methods)

    def _check_salt_bridges(self, l_ions, r_ions):
        ints = []
        for l in l_ions:
            for r in r_ions:
                dist = np.linalg.norm(l['pos'] - r['pos'])
                if dist < 4.0: # Salt bridge cutoff
                    res_info = self._get_res_str(self.receptor, r['indices'][0])
                    ints.append({
                        'residue': res_info,
                        'interaction': "Salt Bridge",
                        'distance': float(dist),
                        'energy': -3.2 # Typical strong interaction
                    })
        return ints

    def _check_hbonds(self, donors, acceptors, direction):
        ints = []
        for d in donors:
            for a in acceptors:
                dist = np.linalg.norm(d['pos'] - a['pos'])
                if dist < 3.5: # 3.5A cutoff for H-bond (heavy atom dist)
                    sub_type = "H-bond (Weak)"
                    energy = -1.0
                    if dist < 2.8: 
                        sub_type = "H-bond (Strong)"
                        energy = -2.5
                    elif dist < 3.2:
                        sub_type = "H-bond"
                        energy = -1.8
                        
                    r_idx = a['indices'][0] if "Ligand->Receptor" == direction else d['indices'][0]
                    res_info = self._get_res_str(self.receptor, r_idx)
                    
                    ints.append({
                        'residue': res_info,
                        'interaction': sub_type,
                        'distance':  float(dist),
                        'energy': energy
                    })
        return ints

    def _check_hydrophobic(self, l_hyd, r_hyd):
        ints = []
        for l in l_hyd:
            for r in r_hyd:
                dist = np.linalg.norm(l['pos'] - r['pos'])
                if dist < 4.5: # Typical cutoff
                    res_info = self._get_res_str(self.receptor, r['indices'][0])
                    ints.append({
                        'residue': res_info,
                        'interaction': "Hydrophobic",
                        'distance': float(dist),
                        'energy': -0.7
                    })
        return ints

    def _check_pi_stacking(self, l_pi, r_pi):
        ints = []
        for l in l_pi:
            for r in r_pi:
                dist = np.linalg.norm(l['pos'] - r['pos'])
                if dist < 5.5:
                    res_info = self._get_res_str(self.receptor, r['indices'][0])
                    ints.append({
                        'residue': res_info,
                        'interaction': "Pi-Stacking",
                        'distance': float(dist),
                        'energy': -2.0
                    })
        return ints

    def _check_metal(self, l_feats, r_metals):
        ints = []
        candidates = l_feats['Donor'] + l_feats['Acceptor'] + l_feats['Halogen']
        for m in r_metals:
            for c in candidates:
                dist = np.linalg.norm(m['pos'] - c['pos'])
                if dist < 2.8:
                    res_info = self._get_res_str(self.receptor, m['indices'][0])
                    ints.append({
                        'residue': res_info,
                        'interaction': "Metal Coordination",
                        'distance': float(dist),
                        'energy': -4.0
                    })
        return ints
    
    def _check_clashes(self):
        ints = []
        # Naive check. Assume Receptor is large, but we only check close ones to Ligand center?
        # Brute force is fine for reporting
        dist = distance_matrix(self.l_pos, self.r_pos) # [N_l, N_r]
        clashes = np.argwhere(dist < 2.0)
        
        for l_idx, r_idx in clashes:
            d = dist[l_idx, r_idx]
            res_info = self._get_res_str(self.receptor, r_idx)
            ints.append({
                'residue': res_info,
                'interaction': "Steric Clash",
                'distance': float(d),
                'energy': 5.0 + (2.0-d)*10
            })
        return ints
        
    def _get_res_str(self, mol, atom_idx):
        atom = mol.GetAtomWithIdx(int(atom_idx))
        info = atom.GetPDBResidueInfo()
        if info:
            return f"{info.GetResidueName()}{info.GetResidueNumber()}"
        return f"UNK{atom_idx}"

