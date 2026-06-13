import uproot
import glob

bg_files = glob.glob('data/jetclass/ZJetsToNuNu_*.root')
sig_files = glob.glob('data/jetclass/HTo*.root')

bg_count = 0
for f in bg_files:
    with uproot.open(f) as file:
        bg_count += file['tree'].num_entries

sig_count = 0
for f in sig_files:
    with uproot.open(f) as file:
        sig_count += file['tree'].num_entries

print(f"Background (ZJets): {bg_count:,}")
print(f"Signal (Higgs): {sig_count:,}")
print(f"Total Jets: {bg_count + sig_count:,}")
