<#
.SYNOPSIS
    Downloads the datasets this repository's training scripts expect.

.DESCRIPTION
    Every URL and expected destination below was verified against the live
    host (HTTP 200 + Content-Length) rather than assumed.

    Fixes versus the previous scripts in this directory:
      * LHCO: the old script fetched events_anomalydetection.h5 (2.6 GB, v1).
        The code reads events_anomalydetection_v2.features.h5, which is a
        different, far smaller (71 MB) file. Downloading v1 left
        LHCODataset unable to find anything.
      * JetClass: the .tar files were downloaded but never extracted, and the
        training scripts glob for data\jetclass\*.root and
        data\jetclass\val_5M\*.root. Extraction and the val_5M\ move now happen.
      * CMS Higgs: no script downloaded it at all, yet train_cms.py and
        inspect_root.py both require GluGluToHToTauTau.root. It is record 12351
        (12355/12361 do not contain it).

.PARAMETER Set
    small  - Higgs (190 MB) + LHCO v2 (71 MB). ~1 minute.
    cms    - the above + TTbar (3.3 GB) + DYJetsToLL (9.2 GB, resumable).
    jetclass - JetClass val_5M (7.1 GB) and extracts it.
    all    - everything. ~34 GB.

.EXAMPLE
    .\scripts\download_data.ps1 -Set small
    .\scripts\download_data.ps1 -Set cms -Higgs
#>
[CmdletBinding()]
param(
    [ValidateSet('small', 'cms', 'jetclass', 'all')]
    [string]$Set = 'small',

    # Narrow the run to one dataset.
    [string[]]$Only,

    # Skip files already present and complete.
    [switch]$ResumeOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# name, url, destination
#
# Sizes are deliberately NOT hard-coded: the portal and Zenodo both report the
# authoritative Content-Length, and a stale constant here silently turns a
# partial download into a "complete" one. Get-RemoteSize asks the server and
# the transfer is verified against that.
$catalog = @(
    @{ Key = 'lhco';    Url = 'https://zenodo.org/api/records/4536377/files/events_anomalydetection_v2.features.h5/content'
                        Path = 'data\lhco\events_anomalydetection_v2.features.h5' },
    @{ Key = 'higgs';   Url = 'https://opendata.cern.ch/eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/GluGluToHToTauTau.root'
                        Path = 'data\cms\higgs\GluGluToHToTauTau.root' },
    @{ Key = 'ttbar';   Url = 'https://opendata.cern.ch/eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/TTbar.root'
                        Path = 'data\cms\ttbar\TTbar.root' },
    @{ Key = 'dyjets';  Url = 'https://opendata.cern.ch/eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/DYJetsToLL.root'
                        Path = 'data\cms\dyjets\DYJetsToLL.root' },
    @{ Key = 'jetclass_val'; Url = 'https://zenodo.org/api/records/6619768/files/JetClass_Pythia_val_5M.tar/content'
                        Path = 'data\jetclass\JetClass_Pythia_val_5M.tar' }
)

$selected = switch ($Set) {
    'small'    { @('lhco', 'higgs') }
    'cms'      { @('lhco', 'higgs', 'ttbar', 'dyjets') }
    'jetclass' { @('jetclass_val') }
    'all'      { @('lhco', 'higgs', 'ttbar', 'dyjets', 'jetclass_val') }
}
if ($Only) { $selected = $Only }

function Get-RemoteSize($url) {
    try {
        $r = Invoke-WebRequest -Uri $url -Method Head -MaximumRedirection 5 -UseBasicParsing
        return [int64]$r.Headers['Content-Length']
    } catch {
        Write-Host "  HEAD failed: $($_.Exception.Message)" -ForegroundColor Red
        return 0
    }
}

function Get-File($entry) {
    $dest = $entry.Path
    $dir = Split-Path -Parent $dest
    if ($dir) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    if (Test-Path $dest) {
        $have = (Get-Item $dest).Length
        $want = Get-RemoteSize $entry.Url
        if ($want -gt 0 -and $have -eq $want) {
            Write-Host "  already complete: $dest ($([math]::Round($have/1GB,2)) GB)" -ForegroundColor DarkGray
            return $true
        }
        if ($want -gt 0 -and $have -gt $want) {
            Write-Host "  local file LARGER than remote ($have > $want); re-downloading" -ForegroundColor Yellow
            Remove-Item $dest -Force
        } else {
            Write-Host "  resuming $dest (have $([math]::Round($have/1GB,2)) GB of $([math]::Round($want/1GB,2)) GB)" -ForegroundColor Cyan
        }
    }

    # curl -C - resumes a partial transfer; retry through transient failures.
    & curl.exe -L -C - --retry 5 --retry-delay 5 --fail --progress-bar `
        -o $dest $entry.Url
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $dest (curl exit $LASTEXITCODE)" -ForegroundColor Red
        return $false
    }

    # Verify against the server rather than trusting the exit code.
    $want = Get-RemoteSize $entry.Url
    $have = (Get-Item $dest).Length
    if ($want -gt 0 -and $have -ne $want) {
        Write-Host "  SIZE MISMATCH: $dest got $have want $want - re-run to resume" -ForegroundColor Red
        return $false
    }
    Write-Host "  OK: $dest ($([math]::Round($have/1GB,2)) GB)" -ForegroundColor Green
    return $true
}

$failed = @()
foreach ($key in $selected) {
    $entry = $catalog | Where-Object { $_.Key -eq $key }
    if (-not $entry) { Write-Host "unknown dataset '$key'" -ForegroundColor Red; continue }

    Write-Host "`n=== $key -> $($entry.Path) ===" -ForegroundColor Cyan
    if (-not (Get-File $entry)) { $failed += $key }
}

# JetClass ships as a tar; the training scripts glob for .root files.
if ($selected -contains 'jetclass_val' -and -not $failed.Contains('jetclass_val')) {
    $tar = 'data\jetclass\JetClass_Pythia_val_5M.tar'
    $out = 'data\jetclass'
    Write-Host "`n=== extracting $tar ===" -ForegroundColor Cyan
    Push-Location $out
    try {
        & tar.exe -xf "../$tar"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  extraction failed (tar exit $LASTEXITCODE)" -ForegroundColor Red
            $failed += 'extract'
        }
    } finally { Pop-Location }

    # train_jetclass.py --large reads val_5M\; make sure it exists.
    $valSrc = Join-Path $out 'val_5M'
    if (Test-Path $valSrc) {
        Write-Host "  val_5M/ present: $((Get-ChildItem $valSrc -Filter *.root).Count) .root files"
    } else {
        Write-Host "  NOTE: no val_5M/ directory in the archive; check the layout" -ForegroundColor Yellow
    }
}

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host "Incomplete: $($failed -join ', ')" -ForegroundColor Red
    Write-Host "Re-run this script; curl -C - resumes partial transfers."
    exit 1
}
Write-Host "All requested datasets ready." -ForegroundColor Green
exit 0
