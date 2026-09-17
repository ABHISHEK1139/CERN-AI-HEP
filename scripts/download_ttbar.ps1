$ErrorActionPreference = "Stop"

Write-Host "Downloading CMS TTbar fresh..."
$dest = "data\cms\ttbar\TTbar.root"
New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
# NOTE: downloads to a temp file first so an interrupted run never leaves a
# corrupt TTbar.root behind; the old file is only replaced on success.
$tmp = "$dest.partial"
Remove-Item -Path $tmp -ErrorAction SilentlyContinue
curl.exe -L --retry 5 --retry-delay 10 -o $tmp "https://opendata.cern.ch/record/12354/files/TTbar.root"
if ($LASTEXITCODE -ne 0) {
    Write-Host "TTbar download failed (curl exit $LASTEXITCODE)."
    exit 1
}
Move-Item -Path $tmp -Destination $dest -Force
Write-Host "`nTTbar download complete!"
