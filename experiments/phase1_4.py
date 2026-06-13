import uproot
import awkward as ak
import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# Create directories
os.makedirs("research", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=== Phase 1: Verify Datasets ===")

# 1. CMS Higgs
cms_file = "data/cms/higgs/GluGluToHToTauTau.root"
if os.path.exists(cms_file):
    with uproot.open(cms_file) as f:
        tree = f["Events"]
        print("\nCMS NanoAOD Branches (Subset):")
        keys = tree.keys()
        for k in ['nMuon', 'nTau', 'nJet', 'MET_pt']:
            if k in keys: print(f" - {k}")
        
        print("\nExtracting first event...")
        # Get first event data
        muon_pt = tree["Muon_pt"].array(entry_stop=1)[0]
        muon_eta = tree["Muon_eta"].array(entry_stop=1)[0]
        muon_phi = tree["Muon_phi"].array(entry_stop=1)[0]
        
        jet_pt = tree["Jet_pt"].array(entry_stop=1)[0]
        jet_eta = tree["Jet_eta"].array(entry_stop=1)[0]
        jet_phi = tree["Jet_phi"].array(entry_stop=1)[0]
        
        print(f"Found {len(muon_pt)} Muons and {len(jet_pt)} Jets in Event 0.")
        
        # Build Graph
        print("\n=== Phase 3: Create Event Graph ===")
        G = nx.Graph()
        
        # Add Muons
        for i in range(len(muon_pt)):
            G.add_node(f"Muon_{i}", pt=muon_pt[i], eta=muon_eta[i], phi=muon_phi[i], type='Muon', color='red')
            
        # Add Jets
        for i in range(len(jet_pt)):
            if jet_pt[i] > 20: # Only plot jets > 20 GeV to avoid clutter
                G.add_node(f"Jet_{i}", pt=jet_pt[i], eta=jet_eta[i], phi=jet_phi[i], type='Jet', color='blue')
                
        # Connect nodes based on Delta R (spatial distance)
        nodes = list(G.nodes(data=True))
        for i, (n1, d1) in enumerate(nodes):
            for j, (n2, d2) in enumerate(nodes):
                if i < j:
                    dEta = d1['eta'] - d2['eta']
                    dPhi = d1['phi'] - d2['phi']
                    # Handle phi periodicity
                    while dPhi > np.pi: dPhi -= 2*np.pi
                    while dPhi < -np.pi: dPhi += 2*np.pi
                    dR = np.sqrt(dEta**2 + dPhi**2)
                    
                    # Connect if Delta R < 2.0 (arbitrary threshold for visualization)
                    if dR < 2.0:
                        G.add_edge(n1, n2, weight=1.0/dR)
                        
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        print("\n=== Phase 4: Visualize Graph ===")
        plt.figure(figsize=(10, 8))
        colors = [data['color'] for _, data in G.nodes(data=True)]
        sizes = [data['pt'] * 10 for _, data in G.nodes(data=True)]
        
        pos = nx.spring_layout(G, k=0.5, weight='weight')
        nx.draw(G, pos, node_color=colors, node_size=sizes, with_labels=True, 
                font_size=8, font_color='white', edge_color='gray', alpha=0.8)
        
        plt.title("LHC Event Graph (CMS NanoAOD)\nRed: Muon, Blue: Jet (Size = pT)")
        plt.savefig("results/event_graph.png", dpi=300, bbox_inches='tight')
        print("Saved visualization to results/event_graph.png")
else:
    print(f"File not found: {cms_file}")

# 2. LHCO Features
lhco_file = "data/lhco/events_anomalydetection_v2.features.h5"
if os.path.exists(lhco_file):
    print("\nLHCO H5 File Keys:")
    with h5py.File(lhco_file, 'r') as f:
        print(list(f.keys()))
else:
    print(f"File not found: {lhco_file}")

print("\n=== Phase 2: Create Dataset Documentation ===")
md_content = """# Dataset Analysis

## 1. CMS Higgs (GluGluToHToTauTau)
- **Events**: ~477,000
- **Particles**: Muons, Taus, Jets, MET, Electrons, Photons
- **Goal**: Real CERN Validation and Event Graph construction.

## 2. LHCO 2020 R&D Dataset
- **Events**: 1.1M (1M background, 100k signal)
- **Goal**: Primary Anomaly Detection Benchmark (Truth Labels included).

## 3. JetClass
- **Goal**: State-of-the-art GNN Benchmark (to be downloaded).
"""
with open("research/dataset_analysis.md", "w") as f:
    f.write(md_content)
print("Saved research/dataset_analysis.md")
