import uproot
file = uproot.open('data/jetclass/ZJetsToNuNu_000.root')
tree = file['tree']

array = tree['part_px'].array()
print(f"Shape of part_px:", array.type)
