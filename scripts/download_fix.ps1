$ErrorActionPreference = "Stop"

Write-Host "Resuming downloads..."
Write-Host "(If your network needs it, add --ssl-no-revoke to the curl commands below.)"

# Ensure directories exist
New-Item -ItemType Directory -Force -Path "data\cms\ttbar" | Out-Null
New-Item -ItemType Directory -Force -Path "data\cms\dyjets" | Out-Null
New-Item -ItemType Directory -Force -Path "data\jetclass" | Out-Null

$files = @(
    @("data\cms\ttbar\TTbar.root", "https://opendata.cern.ch/record/12354/files/TTbar.root"),
    @("data\jetclass\JetClass_Pythia_val_5M.tar", "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_val_5M.tar/content"),
    @("data\cms\dyjets\DYJetsToLL.root", "https://opendata.cern.ch/record/12353/files/DYJetsToLL.root"),
    @("data\jetclass\JetClass_Pythia_train_100M_part0.tar", "https://zenodo.org/api/records/6619768/files/JetClass_Pythia_train_100M_part0.tar/content")
)

$failed = @()
for ($i = 0; $i -lt $files.Count; $i++) {
    $dest, $url = $files[$i]
    Write-Host "`n[$($i + 1)/$($files.Count)] Resuming $dest ..."
    curl.exe -L -C - --retry 5 --retry-delay 10 -o $dest $url
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $dest (curl exit $LASTEXITCODE)"
        $failed += $dest
    }
}

if ($failed.Count -gt 0) {
    Write-Host "`nIncomplete downloads:"
    $failed | ForEach-Object { Write-Host " - $_" }
    Write-Host "Only re-run after all files verify; partial files resume with curl -C -."
    exit 1
}
Write-Host "`nAll downloads finished without curl errors. Verify sizes with scripts/download_monitor.ps1."
