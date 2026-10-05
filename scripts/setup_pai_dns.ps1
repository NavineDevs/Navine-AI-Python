$ErrorActionPreference = "Stop"
$tunnelName = "navine-ai"
$tunnelId = "8327019a-e779-472f-ab47-bf334648483d"
$tunnelTarget = "$tunnelId.cfargotunnel.com"
$hostname = "pai.navinecord.dev"
$certPath = Join-Path $env:USERPROFILE ".cloudflared\cert.pem"

function Test-PaiDns {
    try {
        Resolve-DnsName $hostname -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-CloudflaredCert {
    if (-not (Test-Path $certPath)) { return $false }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $probe = & cloudflared tunnel route dns list $tunnelName 2>&1 | Out-String
    $ErrorActionPreference = $prev
    if ($probe -match "Authentication error") { return $false }
    if ($LASTEXITCODE -ne 0 -and $probe -match "Failed") { return $false }
    return $true
}

Write-Host ""
Write-Host "pai.navinecord.dev setup"
Write-Host "========================"
Write-Host ""

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
    exit 1
}

if (Test-PaiDns) {
    Write-Host "DNS OK: $hostname resolves."
    try {
        $code = (Invoke-WebRequest -Uri "https://$hostname/api/health" -UseBasicParsing -TimeoutSec 20).StatusCode
        Write-Host "Public health: HTTP $code"
    } catch {
        Write-Host "DNS resolves but HTTPS check failed: $($_.Exception.Message)"
        Write-Host "Ensure cloudflared tunnel is running (scripts\start_public_stack.ps1)."
    }
    exit 0
}

Write-Host "Missing DNS for $hostname (NXDOMAIN)."
Write-Host ""

if (-not (Test-CloudflaredCert)) {
    if (Test-Path $certPath) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        Rename-Item $certPath "$certPath.stale-$stamp" -Force
        Write-Host "Stale cert.pem moved aside. Refreshing Cloudflare login..."
    } else {
        Write-Host "No cert.pem found. Cloudflare login required for CLI DNS route."
    }
    Write-Host ""
    Write-Host "A browser window will open. Log into Cloudflare and pick zone: navinecord.dev"
    Write-Host ""
    & cloudflared tunnel login
    if (-not (Test-Path $certPath)) {
        Write-Host ""
        Write-Host "Login did not create cert.pem. Add DNS manually:"
        Write-Host "  CNAME  pai  ->  $tunnelTarget  (Proxied ON)"
        exit 1
    }
}

Write-Host "Creating tunnel DNS route..."
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& cloudflared tunnel route dns $tunnelName $hostname 2>&1 | Write-Host
$ErrorActionPreference = $prev
Start-Sleep -Seconds 3

if (Test-PaiDns) {
    Write-Host "OK: $hostname DNS created."
    Write-Host "Verify: curl https://$hostname/api/health"
    exit 0
}

Write-Host ""
Write-Host "CLI route failed. Add this record in Cloudflare Dashboard:"
Write-Host "  Type: CNAME | Name: pai | Target: $tunnelTarget | Proxy: ON"
Write-Host ""
exit 1
