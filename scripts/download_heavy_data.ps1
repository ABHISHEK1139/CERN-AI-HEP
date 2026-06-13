Write-Host "Starting sequential downloads with curl to avoid memory buffering issues..."

Write-Host "`n[1/5] Downloading LHCO raw (2.6 GB)..."
curl.exe -L -C - -o "data\lhco\raw\events_anomalydetection.h5" "https://zenodo.org/api/records/4536377/files/events_anomalydetection.h5/content"

Write-Host "`n[2/5] Downloading CMS TTbar (3.3 GB)..."
curl.exe -L -C - -o "data\cms\ttbar\TTbar.root" "https://opendata.cern.ch/record/12354/files/TTbar.root"

Write-Host "`n[3/5] Downloading JetClass val (7.1 GB)..."
curl.exe -L -C - -o "data\jetclass\JetClass_Pythia_val_5M.tar" "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_val_5M.tar/content"

Write-Host "`n[4/5] Downloading CMS DYJetsToLL (8.6 GB)..."
curl.exe -L -C - -o "data\cms\dyjets\DYJetsToLL.root" "https://opendata.cern.ch/record/12353/files/DYJetsToLL.root"

Write-Host "`n[5/5] Downloading JetClass train part0 (14.1 GB)..."
curl.exe -L -C - -o "data\jetclass\JetClass_Pythia_train_100M_part0.tar" "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_train_100M_part0.tar/content"

Write-Host "`nAll downloads complete!"
