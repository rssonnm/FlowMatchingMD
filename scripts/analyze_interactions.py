import argparse
import pandas as pd
from rdkit import Chem
from fmmd.analysis import InteractionAnalyzer

def run_analysis(receptor_path, ligand_path, output_csv):
    print(f"Analyzing Interactions for: {ligand_path}")
    print(f"Receptor: {receptor_path}")
    
    # Load molecules
    # Receptor might be PDB or PDBQT. RDKit prefers PDB.
    # If PDBQT, we might need to convert or just try parsing.
    # Warning: PDBQT parsing in RDKit is tricky. Assume we use the PDB version if available for analysis
    # or rely on basic PDB parser if file is PDB.
    
    if receptor_path.endswith('pdbqt'):
        print("Warning: RDKit might not parse PDBQT perfectly for connectivity. PDB is preferred.")
    
    receptor = Chem.MolFromPDBFile(receptor_path, removeHs=False)
    if receptor is None:
        print(f"Error: Could not load receptor {receptor_path}")
        return

    ligand = Chem.SDMolSupplier(ligand_path, removeHs=False)[0]
    if ligand is None:
        print(f"Error: Could not load ligand {ligand_path}")
        return
        
    # Analyze
    analyzer = InteractionAnalyzer(receptor, ligand)
    interactions = analyzer.analyze()
    
    # Create DataFrame
    if not interactions:
        print("No specific interactions detected.")
        return
        
    df = pd.DataFrame(interactions)
    
    # Sort by Energy (strongest first)
    df = df.sort_values(by='energy', ascending=True)
    
    # Display Table
    print("\n" + "="*60)
    print(f"{'RESIDUE':<10} | {'INTERACTION':<20} | {'DIST (Å)':<8} | {'ENERGY':<10}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['residue']:<10} | {row['interaction']:<20} | {row['distance']:.2f}     | {row['energy']:.2f}")
    print("="*60 + "\n")
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Detailed report saved to {output_csv}")
    
    # Summary
    print("Summary:")
    summary = df['interaction'].value_counts()
    for k, v in summary.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--output", default="results/interaction_report.csv")
    
    args = parser.parse_args()
    
    run_analysis(args.receptor, args.ligand, args.output)
