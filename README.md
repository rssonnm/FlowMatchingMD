# FMMD: Flow Matching for Molecular Docking

A state-of-the-art molecular docking library using **Riemannian Flow Matching on SE(3)** manifolds. This project leverages geometric deep learning and physics-informed constraints to generate high-quality ligand binding poses.

## Features

- **SE(3) Equivariant Backbone**: Uses `InteractionAwareEquivariantBlock` to respect molecular geometry and rotation/translation symmetries.
- **Riemannian Flow Matching**: Solves ODEs on the SE(3) manifold for natural rigid-body motion validation.
- **Physics-Informed Refinement**: Integrates **Langevin Dynamics** to minimize steric clashes and optimize binding energy post-generation.
- **Hardware Acceleration**: Optimized for **Apple Silicon (M1/M2/M3/M4)** via Metal Performance Shaders (MPS) and NVIDIA CUDA.
- **Interaction Analysis**: Built-in tools to detect Hydrogen Bonds, Salt Bridges, Pi-Stacking, and more.

## Installation

```bash
git https://github.com/rssonnm/FlowMatchingMD.git
cd FlowMatchingMD
pip install -e .
```

### Requirements
- Python 3.9+
- PyTorch (2.0+)
- RDKit
- SciPy
- Pandas, Matplotlib

```bash
pip install -r requirements.txt
```

## Usage

### 1. Molecular Docking
Run the optimized docking pipeline (supports batch sampling and refinement):

```bash
python scripts/run_docking.py \
  --receptor data/receptor.pdb \
  --ligands data/ligand.sdf \
  --output results/docked_pose.sdf \
  --device mps \
  --samples 40 \
  --steps 20
```

### 2. Interaction Analysis
Generate a detailed report of chemical interactions:

```bash
python scripts/analyze_interactions.py \
  --receptor data/receptor.pdb \
  --ligand results/docked_pose.sdf \
  --output results/interaction_report.csv
```

### 3. Visualization
Visualize the results and compare with ground truth (if available):

```bash
python scripts/visualize_results.py \
  --receptor data/receptor.pdb \
  --ligands data/ligand.sdf results/docked_pose.sdf \
  --labels "Ground Truth" "Docked Pose" \
  --output results/plot.png
```

## Project Structure

- `src/fmmd/`: Core library code.
- `scripts/`: Executable scripts for docking and analysis.
- `data/`: Example input files.
- `results/`: Output directory.

## License
[MIT License](LICENSE)
