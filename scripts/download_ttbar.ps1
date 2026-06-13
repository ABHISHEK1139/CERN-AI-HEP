Write-Host "Downloading CMS TTbar fresh (server does not support resume)..."
Remove-Item -Path "data\cms\ttbar\TTbar.root" -ErrorAction SilentlyContinue
curl.exe -L --ssl-no-revoke --retry 5 --retry-delay 10 -o "data\cms\ttbar\TTbar.root" "https://opendata.cern.ch/record/12354/files/TTbar.root"
Write-Host "`nTTbar download complete!"
