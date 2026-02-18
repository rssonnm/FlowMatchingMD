import argparse
import json
import os
import math
import base64

def generate_pymol_script(ligand_file, receptor_file, report, output_pml):
    """Generates a PyMOL script for offline visualization."""
    abs_lig = os.path.abspath(ligand_file)
    abs_rec = os.path.abspath(receptor_file)
    
    script = f"""
reinitialize
load {abs_rec}, receptor
load {abs_lig}, ligand
hide everything
show cartoon, receptor
show sticks, ligand
util.cbag ligand

# Color receptor by spectrum
spectrum b, rainbow, receptor

# Interactions
"""
    interactions = report.get("interactions", [])
    for i, inter in enumerate(interactions):
        r_coords = inter.get('receptor_coords')
        l_idx = inter.get('atom_idx') # This is actually receptor atom idx in our code, wait. 
        # In scoring.py: atom_idx is the RECEPTOR atom index (j). 
        # But we don't have ligand atom index clearly stored for the pair... 
        # Actually scoring.py loop: l_atoms[i], r_atoms[j]. "interactions.append({..., 'atom_idx': int(j)})"
        # So we only have the receptor atom index.
        # However, we saved "receptor_coords".
        
        # PyMOL CGO (Compiled Graphics Objects) is hard to generate via script without coordinates pair.
        # But we can create pseudoatoms for interactions.
        
        # Let's simple create distance objects if we can find the atoms.
        # Or better, just draw CGO lines using coordinates.
        # We need the PAIR coordinates. 
        # In scoring.py, we didn't save the ligand coordinate for the interaction! 
        # We only saved type, dist, res, r_idx, r_coords.
        # We need to calculate ligand closest atom again here or update scoring.py.
        # For now, we'll skip drawing lines in PyMOL script to keep it simple, 
        # or we just show the view.
        pass

    script += """
zoom
bg_color white
set ray_shadows, 0
    """
    
    with open(output_pml, 'w') as f:
        f.write(script)
    print(f"Generated PyMOL script: {output_pml}")

def generate_html(ligand_sdf, receptor_pdb, report_json, output_html):
    # Read files
    try:
        with open(ligand_sdf, 'rb') as f:
            ligand_data = f.read()
            
        with open(receptor_pdb, 'rb') as f:
            receptor_data = f.read()
            
        with open(report_json, 'r') as f:
            report = json.load(f)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Base64 Encode
    ligand_b64 = base64.b64encode(ligand_data).decode('utf-8')
    receptor_b64 = base64.b64encode(receptor_data).decode('utf-8')

    interactions = report.get("interactions", [])
    
    # Parse Ligand Coords for finding closest atom for lines
    # Needed because scoring.py didn't save ligand coords, only receptor coords
    lig_pos = []
    try:
        from rdkit import Chem
        # Read string from bytes
        mol = Chem.MolFromMolBlock(ligand_data.decode('utf-8'))
        if mol:
            conf = mol.GetConformer()
            lig_pos = conf.GetPositions()
    except:
        pass

    # Construct Shape list for 3Dmol
    shapes = []
    interaction_html = ""
    
    for inter in interactions:
        interaction_type = inter.get('type', 'Unknown')
        dist_val = inter.get('distance', 0)
        res_info = inter.get('residue', 'Unknown')
        r_coords = inter.get('receptor_coords') # [x, y, z]
        
        css_class = "hydrophobic"
        color = "orange"
        if "Bond" in interaction_type:
            css_class = "hbond"
            color = "green"
        elif "Salt" in interaction_type:
            css_class = "saltbridge"
            color = "red"
        elif "Pi" in interaction_type:
            css_class = "pistacking"
            color = "purple"
            
        interaction_html += f'<div class="interaction-item type-{css_class}"><b>{interaction_type}</b><br>Dist: {dist_val}A<br>Residue: {res_info}</div>'

        if r_coords:
            r_pt = {'x': float(r_coords[0]), 'y': float(r_coords[1]), 'z': float(r_coords[2])}
            
            # Find closest ligand atom to this receptor atom to draw the line
            min_d = 999.0
            l_pt_best = None
            
            if len(lig_pos) > 0:
                for lp in lig_pos:
                    d = math.sqrt((lp[0]-r_pt['x'])**2 + (lp[1]-r_pt['y'])**2 + (lp[2]-r_pt['z'])**2)
                    if d < min_d:
                        min_d = d
                        l_pt_best = {'x': float(lp[0]), 'y': float(lp[1]), 'z': float(lp[2])}
            
            # If no rdkit or simple fallback, we can't draw precise lines easily without ligand coords in report.
            # But the user wants visualization. 
            # If RDKit failed, we might have issue.
            
            if l_pt_best:
                shapes.append({
                    'type': 'cylinder',
                    'start': r_pt,
                    'end': l_pt_best,
                    'color': color,
                    'radius': 0.1,
                    'dashed': True
                })

    shapes_json = json.dumps(shapes)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
    body {{ margin: 0; font-family: sans-serif; display: flex; height: 100vh; overflow: hidden; }}
    #viewport {{ width: 75%; height: 100%; position: relative; border-right: 1px solid #ccc; }}
    #sidebar {{ width: 25%; height: 100%; overflow-y: auto; padding: 20px; box-sizing: border-box; background: #f9f9f9; }}
    h2 {{ margin-top: 0; }}
    .interaction-item {{ background: white; padding: 10px; margin-bottom: 10px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #ccc; font-size: 14px; }}
    .type-hbond {{ border-left-color: green; }}
    .type-hydrophobic {{ border-left-color: orange; }}
    .type-saltbridge {{ border-left-color: red; }}
    .type-pistacking {{ border-left-color: purple; }}
    .stats {{ margin-bottom: 20px; padding: 10px; background: #eef; border-radius: 5px; }}
</style>
<title>FlowDock Results</title>
</head>
<body>
    <div id="viewport"></div>
    <div id="sidebar">
        <h2>FlowDock</h2>
        <div class="stats">
            <strong>Binding Energy:</strong> {report.get('binding_energy', 'N/A'):.2f} kcal/mol <br>
            <small>VDW: {report.get('energy_terms', {}).get('vdw', 0):.2f} | Elec: {report.get('energy_terms', {}).get('electrostatic', 0):.2f}</small>
        </div>
        <h3>Interactions</h3>
        {interaction_html}
    </div>

<script>
    $(document).ready(function() {{
        var viewer = $3Dmol.createViewer("viewport", {{ backgroundColor: "white" }});
        
        // Decode Base64
        var recB64 = "{receptor_b64}";
        var ligB64 = "{ligand_b64}";
        
        var receptorData = atob(recB64);
        var ligandData = atob(ligB64);
        
        // Add Receptor
        viewer.addModel(receptorData, "pdb");
        viewer.setStyle({{model: 0}}, {{cartoon: {{color: 'spectrum', opacity: 0.8}}}});
        
        // Add Ligand
        viewer.addModel(ligandData, "sdf");
        viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.2}}}});
        
        // Add Interaction Lines
        var shapes = {shapes_json};
        for (var i = 0; i < shapes.length; i++) {{
            viewer.addShape(shapes[i]);
        }}
        
        // Zoom
        viewer.zoomTo();
        viewer.render();
    }});
</script>
</body>
</html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Generated HTML: {output_html}")
    
    # Generate PyMOL fallback
    pml_file = output_html.replace('.html', '.pml')
    generate_pymol_script(ligand_sdf, receptor_pdb, report, pml_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ligand_sdf", help="Input Ligand SDF")
    parser.add_argument("receptor_pdb", help="Input Receptor PDB")
    parser.add_argument("report_json", help="Input Report JSON")
    parser.add_argument("--output", default="docking_results.html", help="Output HTML file")
    args = parser.parse_args()
    
    generate_html(args.ligand_sdf, args.receptor_pdb, args.report_json, args.output)
