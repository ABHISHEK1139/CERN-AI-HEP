Write-Host "Resuming downloads with SSL fix..."

# Ensure directories exist
New-Item -ItemType Directory -Force -Path "data\cms\ttbar" | Out-Null
New-Item -ItemType Directory -Force -Path "data\cms\dyjets" | Out-Null
New-Item -ItemType Directory -Force -Path "data\jetclass" | Out-Null

Write-Host "`n[1/4] Resuming CMS TTbar (3.3 GB)..."
curl.exe -L -C - --ssl-no-revoke --retry 5 --retry-delay 10 -o "data\cms\ttbar\TTbar.root" "https://opendata.cern.ch/record/12354/files/TTbar.root"

Write-Host "`n[2/4] Resuming JetClass val (7.1 GB)..."
curl.exe -L -C - --ssl-no-revoke --retry 5 --retry-delay 10 -o "data\jetclass\JetClass_Pythia_val_5M.tar" "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_val_5M.tar/content"

Write-Host "`n[3/4] Downloading CMS DYJetsToLL (8.6 GB)..."
curl.exe -L -C - --ssl-no-revoke --retry 5 --retry-delay 10 -o "data\cms\dyjets\DYJetsToLL.root" "https://opendata.cern.ch/record/12353/files/DYJetsToLL.root"

Write-Host "`n[4/4] Downloading JetClass train part0 (14.1 GB)..."
curl.exe -L -C - --ssl-no-revoke --retry 5 --retry-delay 10 -o "data\jetclass\JetClass_Pythia_train_100M_part0.tar" "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_train_100M_part0.tar/content"

Write-Host "`nAll downloads complete!"
