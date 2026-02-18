import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from rdkit import Chem

def parse_pdb_backbone(pdb_file):
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except: pass
    return np.array(coords)

def parse_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    if not suppl: return None
    mol = suppl[0]
    if not mol: return None
    return mol.GetConformer().GetPositions()

def calculate_rmsd_coords(coords_ref, coords_pred):
    if coords_ref is None or coords_pred is None: return 0.0
    if len(coords_ref) != len(coords_pred): return 0.0
    return np.sqrt(np.mean(np.sum((coords_ref - coords_pred)**2, axis=1)))

def visualize_comparison(receptor_file, ligand_files, labels=None, output_file="comparison_plot.png"):
    rec_coords = parse_pdb_backbone(receptor_file)
    
    ligands_coords = []
    for f in ligand_files:
        ligands_coords.append(parse_sdf(f))
        
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(ligand_files))]
        labels[0] = "Ground Truth"
        
    # Calculate RMSDs
    rmsds = []
    ref_coords = ligands_coords[0]
    for i, coords in enumerate(ligands_coords):
        if i == 0:
            rmsds.append(0.0)
        else:
            rmsds.append(calculate_rmsd_coords(ref_coords, coords))

    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Receptor (Context)
    # Filter receptor atoms near the first ligand to avoid clutter
    if len(rec_coords) > 0 and ref_coords is not None:
        center = ref_coords.mean(axis=0)
        dists = np.linalg.norm(rec_coords - center, axis=1)
        mask = dists < 20.0
        rec_plot = rec_coords[mask]
        ax.scatter(rec_plot[:, 0], rec_plot[:, 1], rec_plot[:, 2], 
                   c='lightgray', alpha=0.5, s=15, label='Receptor (Backbone)')
        
    # 2. Plot Ligands
    colors = ['green', 'red', 'blue', 'orange', 'purple']
    
    for i, coords in enumerate(ligands_coords):
        if coords is None: continue
        
        label_text = f"{labels[i]}"
        if i > 0:
            label_text += f" (RMSD: {rmsds[i]:.2f}Å)"
            
        c = colors[i % len(colors)]
        
        # Plot atoms
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                   c=c, s=50, alpha=0.8, label=label_text, depthshade=False)
        
        # Draw bonds (simple distance based)
        # N^2 check is fine for small ligands
        num_atoms = len(coords)
        for j in range(num_atoms):
            for k in range(j+1, num_atoms):
                dist = np.linalg.norm(coords[j] - coords[k])
                if dist < 1.6: # Bond threshold
                    ax.plot([coords[j,0], coords[k,0]], 
                            [coords[j,1], coords[k,1]], 
                            [coords[j,2], coords[k,2]], c=c, linewidth=2)

    ax.set_title("SOTA Docking Comparison", fontsize=16)
    ax.legend(loc='upper right')
    
    # Auto scale view
    all_points = np.vstack([c for c in ligands_coords if c is not None])
    if len(all_points) > 0:
        center = all_points.mean(axis=0)
        max_range = 10.0 # View radius
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"Saved comparison plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--ligands", required=True, nargs='+')
    parser.add_argument("--labels", nargs='+')
    parser.add_argument("--output", default="comparison_plot.png")
    
    args = parser.parse_args()
    visualize_comparison(args.receptor, args.ligands, args.labels, args.output)
