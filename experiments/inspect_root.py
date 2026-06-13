import uproot

path = "data/cms/higgs/GluGluToHToTauTau.root"
with uproot.open(path) as file:
    print("Keys in ROOT file:", file.keys())
    # Find the main tree
    tree_key = None
    for k in file.keys():
        if "Events" in k:
            tree_key = k
            break
            
    if tree_key:
        tree = file[tree_key]
        print(f"\nTree '{tree_key}' has {tree.num_entries} events.")
        print("\nRelevant Jet properties:")
        for k in tree.keys():
            if k.startswith("Jet_pt") or k.startswith("Jet_eta") or k.startswith("Jet_phi") or k.startswith("Jet_mass"):
                print(" -", k)
