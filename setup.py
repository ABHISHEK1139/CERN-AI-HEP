"""Setup script for CERN-AI project."""

from setuptools import setup, find_packages

setup(
    name="CERN-AI-HEP",
    version="0.1.0",
    description="Graph Neural Network Based Anomaly Detection for LHC Events",
    author="Abhishek",
    author_email="ak612520208365@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torch-geometric>=2.4",
        "uproot>=5.0",
        "awkward>=2.0",
        "h5py>=3.0",
        "tables>=3.8",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "Pillow>=10.0",
        "scikit-learn>=1.3",
        "mlflow>=2.8",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        "requests>=2.31",
        "streamlit>=1.30.0",
        "networkx>=3.0",
        "umap-learn>=0.5",
    ],
)
