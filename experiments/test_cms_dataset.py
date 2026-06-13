import sys
from graph_builder.cms_dataset import CMSDataset

def test():
    print("Testing CMSDataset...")
    # Load just 100 events from Higgs dataset
    dataset = CMSDataset(
        root="data/cms/graphs", 
        root_file_path="data/cms/higgs/GluGluToHToTauTau.root", 
        label=1, 
        sample_size=100
    )
    
    print(f"Loaded {len(dataset)} graphs.")
    if len(dataset) > 0:
        graph = dataset[0]
        print(f"Graph 0: x={graph.x.shape}, edge_index={graph.edge_index.shape}, y={graph.y}")
        print("x:", graph.x)

if __name__ == "__main__":
    test()
