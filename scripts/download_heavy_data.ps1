$ErrorActionPreference = "Stop"

Write-Host "Starting sequential downloads with curl to avoid memory buffering issues..."

$files = @(
    @("data\lhco\raw\events_anomalydetection.h5", "https://zenodo.org/api/records/4536377/files/events_anomalydetection.h5/content"),
    @("data\cms\ttbar\TTbar.root", "https://opendata.cern.ch/record/12354/files/TTbar.root"),
    @("data\jetclass\JetClass_Pythia_val_5M.tar", "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_val_5M.tar/content"),
    @("data\cms\dyjets\DYJetsToLL.root", "https://opendata.cern.ch/record/12353/files/DYJetsToLL.root"),
    @("data\jetclass\JetClass_Pythia_train_100M_part0.tar", "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_train_100M_part0.tar/content")
)

$failed = @()
for ($i = 0; $i -lt $files.Count; $i++) {
    $dest, $url = $files[$i]
    Write-Host "`n[$($i + 1)/$($files.Count)] Downloading $dest..."
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    curl.exe -L -C - --retry 5 --retry-delay 10 -o $dest $url
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $dest (curl exit $LASTEXITCODE)"
        $failed += $dest
    }
}

if ($failed.Count -gt 0) {
    Write-Host "`nFailed downloads:"
    $failed | ForEach-Object { Write-Host " - $_" }
    Write-Host "Re-run this script to resume (curl -C - continues partial files)."
    exit 1
}
Write-Host "`nAll downloads complete!"
