"""Setup script for CERN-AI project."""

from setuptools import setup, find_packages

setup(
    name="cern-ai",
    version="0.1.0",
    description="Graph Neural Network Based Anomaly Detection for LHC Events",
    author="Abhishek",
    author_email="ak612520208365@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0",
        "torch-geometric>=2.4",
        "uproot>=5.0",
        "awkward>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "scikit-learn>=1.3",
        "mlflow>=2.8",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
)
