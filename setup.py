from setuptools import setup, find_packages

setup(
    name="fmmd",
    version="1.0.0",
    description="Flow Matching Molecular Docking: State-of-the-art molecular docking using Optimal Transport and Cross-Attention EGNN.",
    author="FMMD Team", # Neutral author as requested
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=2.0.0",
        "rdkit",
        "matplotlib" # For visualization scripts
    ],
    entry_points={
        "console_scripts": [
            "fmmd-dock=scripts.run_docking:main",
        ],
    },
)
