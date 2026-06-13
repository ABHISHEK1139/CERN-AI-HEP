$downloads = @(
    @{ Name = "CMS TTbar";      Path = "data\cms\ttbar\TTbar.root";                          Expected = 3521093632 }
    @{ Name = "JetClass Val";   Path = "data\jetclass\JetClass_Pythia_val_5M.tar";            Expected = 7592034304 }
    @{ Name = "CMS DYJets";    Path = "data\cms\dyjets\DYJetsToLL.root";                     Expected = 9231532032 }
    @{ Name = "JetClass Train"; Path = "data\jetclass\JetClass_Pythia_train_100M_part0.tar";  Expected = 15141838848 }
)

$completed = @(
    @{ Name = "CMS Higgs"; Path = "data\cms\higgs\GluGluToHToTauTau.root" }
    @{ Name = "LHCO Feat"; Path = "data\lhco\events_anomalydetection_v2.features.h5" }
    @{ Name = "LHCO Raw";  Path = "data\lhco\raw\events_anomalydetection.h5" }
)

function Get-Bar {
    param([double]$Pct)
    $filled = [math]::Floor($Pct / 2.5)
    if ($filled -gt 40) { $filled = 40 }
    if ($filled -lt 0) { $filled = 0 }
    $empty = 40 - $filled
    return ("#" * $filled) + ("-" * $empty)
}

while ($true) {
    Clear-Host
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "         CERN-AI Dataset Download Monitor" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "  [COMPLETED]" -ForegroundColor Green
    foreach ($d in $completed) {
        if (Test-Path $d.Path) {
            $sizeMB = [math]::Round((Get-Item $d.Path).Length / 1MB, 1)
            Write-Host ("    [OK] {0,-16} {1,8} MB  [########################################] 100%%" -f $d.Name, $sizeMB) -ForegroundColor Green
        }
    }
    Write-Host ""

    Write-Host "  [DOWNLOADING - 4 Parallel Streams]" -ForegroundColor Yellow
    $totalDown = [long]0
    $totalExp = [long]0

    foreach ($d in $downloads) {
        $current = [long]0
        if (Test-Path $d.Path) {
            $current = (Get-Item $d.Path).Length
        }
        $pct = 0
        if ($d.Expected -gt 0) {
            $pct = [math]::Round(($current / $d.Expected) * 100, 1)
            if ($pct -gt 100) { $pct = 100 }
        }
        $sizeMB = [math]::Round($current / 1MB, 1)
        $expMB = [math]::Round($d.Expected / 1MB, 1)
        $bar = Get-Bar -Pct $pct
        $totalDown += $current
        $totalExp += $d.Expected

        if ($pct -ge 100) {
            $color = "Green"
            $icon = "[OK]"
        } elseif ($pct -gt 0) {
            $color = "Yellow"
            $icon = "[>>]"
        } else {
            $color = "Red"
            $icon = "[..]"
        }
        Write-Host ("    {0} {1,-16} {2,9} / {3,9} MB  [{4}] {5}%%" -f $icon, $d.Name, $sizeMB, $expMB, $bar, $pct) -ForegroundColor $color
    }

    Write-Host ""
    $totalPct = 0
    if ($totalExp -gt 0) {
        $totalPct = [math]::Round(($totalDown / $totalExp) * 100, 1)
    }
    $totalDownGB = [math]::Round($totalDown / 1GB, 2)
    $totalExpGB = [math]::Round($totalExp / 1GB, 2)
    $totalBar = Get-Bar -Pct $totalPct
    Write-Host ("  TOTAL:  {0} GB / {1} GB  [{2}] {3}%%" -f $totalDownGB, $totalExpGB, $totalBar, $totalPct) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Ctrl+C to close monitor (downloads continue in background)" -ForegroundColor DarkGray
    Write-Host "  Refreshing every 5s..." -ForegroundColor DarkGray

    if ($totalPct -ge 100) {
        Write-Host ""
        Write-Host "  ALL DOWNLOADS COMPLETE!" -ForegroundColor Green
        break
    }

    Start-Sleep -Seconds 5
}
