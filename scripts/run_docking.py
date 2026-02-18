
import argparse
import torch
import time
import os
import numpy as np
from rdkit import Chem
from fmmd.model_sota import SE3EquivariantModel
from fmmd.sampler import SE3ODESolver
from fmmd.refine import LangevinRefiner
from fmmd.scoring import PhysicsScoring
from fmmd.data_processing import parse_ligand_with_features, parse_receptor

def run_sota_optimized(receptor_path, ligand_path, output_path, num_samples=8, steps=20, device='cpu', refine_steps=50):
    start_time = time.time()
    
    # 1. Device Setup for Apple Silicon
    if device == 'mps' and torch.backends.mps.is_available():
        print("🚀 Running on Apple Metal (MPS) Acceleration!")
        device = torch.device('mps')
    elif device == 'cuda' and torch.cuda.is_available():
        print("🚀 Running on NVIDIA CUDA!")
        device = torch.device('cuda')
    else:
        print("⚠️  Running on CPU (might be slow)")
        device = torch.device('cpu')
        
    print(f"Receptor: {receptor_path}")
    print(f"Ligand: {ligand_path}")
    
    # 2. Parse Data
    try:
        ligand_data = parse_ligand_with_features(ligand_path)
        receptor_data = parse_receptor(receptor_path)
    except Exception as e:
        print(f"Error parsing data: {e}")
        return

    # Check features dimensions
    node_in_dim = ligand_data.features.shape[1]
    hidden_dim = 64
    
    # 3. Initialize Model
    model = SE3EquivariantModel(node_in_dim=node_in_dim, hidden_dim=hidden_dim)
    model.to(device)
    model.eval()
    
    # 4. Prepare Shared Data
    if receptor_data.features is not None:
        rec_center = torch.mean(receptor_data.positions, dim=0)
    else:
        rec_center = torch.mean(receptor_data.positions, dim=0)
    
    pocket_center = rec_center.to(device)
    
    h_l = ligand_data.features.to(device)
    x_r = receptor_data.positions.to(device)
    
    # Handle receptor features
    if receptor_data.features is not None:
        h_r = receptor_data.features.to(device)
        if h_r.shape[1] < node_in_dim:
            padding = torch.zeros(h_r.shape[0], node_in_dim - h_r.shape[1], device=device)
            h_r = torch.cat([h_r, padding], dim=1)
        elif h_r.shape[1] > node_in_dim:
             h_r = h_r[:, :node_in_dim]
    else:
        h_r = torch.zeros(x_r.shape[0], node_in_dim, device=device)
        
    # 5. Solver Setup
    solver = SE3ODESolver(model, step_size=1.0/steps)
    
    # 6. Sampling Loop (Generating N poses)
    best_score = float('inf')
    best_pose = None
    best_idx = -1
    
    scorer = PhysicsScoring()
    refiner = LangevinRefiner(steps=refine_steps, step_size=0.1, temperature=0.01) # Low temp for minimization
    
    print(f"\nGenerative Sampling ({num_samples} poses)...")
    
    # Prepare Physics Data for Scoring
    l_types = ligand_data.atom_types.to(device)
    r_types = receptor_data.atom_types.to(device)
    l_charges = ligand_data.charges.to(device)
    r_charges = receptor_data.charges.to(device)
    
    # Update MolecularData objects to hold device tensors for Refiner
    ligand_data.atom_types = l_types
    ligand_data.charges = l_charges
    # Position update happens in loop
    
    receptor_data.atom_types = r_types
    receptor_data.charges = r_charges
    receptor_data.positions = x_r # Already on device
    
    for i in range(num_samples):
        # Initial State
        p0 = torch.randn(3, device=device) + pocket_center
        R0 = torch.eye(3, device=device)
        
        # Dock
        R_final, p_final = solver.solve(R_init=R0, p_init=p0, h_l=h_l, h_r=h_r, x_r=x_r, use_repulsion=True)
        
        # Reconstruction
        x_l_local = h_l[:, :3]
        x_pred = torch.matmul(x_l_local, R_final.transpose(-1, -2)) + p_final
        
        # Refinement (Energy Minimization)
        # Fix: Need to update ligand_data positions temporarily for refiner
        # But Refiner takes tensors or data object? 
        # Refiner expects MolecularData objects but modifies positions.
        # Let's create a temporary copy of ligand_data with new positions
        ligand_data.positions = x_pred
        receptor_data.positions = x_r # Ensure on device
        
        # Run Refinement
        x_refined, energies = refiner.refine(ligand_data, receptor_data)
        
        # Score
        final_energy, _ = scorer.calculate_energy(
            x_refined, x_r, l_charges, r_charges, l_types, r_types
        )
        
        print(f"  Pose {i+1}: Energy = {final_energy:.2f} kcal/mol")
        
        if final_energy < best_score:
            best_score = final_energy
            best_pose = x_refined
            best_idx = i
            
    print(f"\n🏆 Best Pose: #{best_idx+1} (Energy: {best_score:.2f} kcal/mol)")
    print(f"Total Time: {time.time() - start_time:.2f}s")
    
    # 7. Save Best Pose
    suppl = Chem.SDMolSupplier(ligand_path)
    mol = suppl[0]
    conf = mol.GetConformer()
    
    x_final_np = best_pose.detach().cpu().numpy()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(x_final_np[i,0]), float(x_final_np[i,1]), float(x_final_np[i,2])))
        
    w = Chem.SDWriter(output_path)
    w.write(mol)
    w.close()
    print(f"Saved Optimized Result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--ligands", required=True, nargs='+')
    parser.add_argument("--output", default="results/sota_optimized_pose.sdf")
    parser.add_argument("--samples", type=int, default=8, help="Number of poses to generate")
    parser.add_argument("--device", default="mps", help="Device (cpu, cuda, mps)")
    parser.add_argument("--steps", type=int, default=20, help="ODE Solver steps")
    
    args = parser.parse_args()
    
    target_ligand = args.ligands[0]
    run_sota_optimized(args.receptor, target_ligand, args.output, 
                       num_samples=args.samples, steps=args.steps, device=args.device)
