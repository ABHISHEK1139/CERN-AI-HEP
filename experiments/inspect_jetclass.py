import uproot

path = "data/jetclass/val_5M/ZJetsToNuNu_120.root"
with uproot.open(path) as file:
    print("Keys:", file.keys())
    tree = file["tree"]
    print(f"\nTree has {tree.num_entries} events.")
    print("Branches:")
    for k in tree.keys():
        print(" -", k)
