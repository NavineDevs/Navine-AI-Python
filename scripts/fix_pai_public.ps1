$ErrorActionPreference = "Stop"
$tunnelId = "8327019a-e779-472f-ab47-bf334648483d"
$tunnelTarget = "$tunnelId.cfargotunnel.com"
$hostname = "pai.navinecord.dev"
$zone = "navinecord.dev"

Write-Host ""
Write-Host "PAI public URL fix"
Write-Host "=================="
Write-Host ""

try {
    Resolve-DnsName $hostname -ErrorAction Stop | Out-Null
    Write-Host "DNS OK: $hostname already resolves."
} catch {
    Write-Host "DNS MISSING: $hostname does not resolve (browser shows site not found)."
    Write-Host ""
    Write-Host "Add this in Cloudflare Dashboard -> navinecord.dev -> DNS:"
    Write-Host "  Type: CNAME"
    Write-Host "  Name: pai"
    Write-Host "  Target: $tunnelTarget"
    Write-Host "  Proxy: ON (orange cloud)"
    Write-Host ""
    if (Get-Command cloudflared -ErrorAction SilentlyContinue) {
        $cert = Join-Path $env:USERPROFILE ".cloudflared\cert.pem"
        if (Test-Path $cert) {
            Write-Host "Trying cloudflared tunnel route dns..."
            cloudflared tunnel route dns navine-ai $hostname
        } else {
            Write-Host "Or run: cloudflared tunnel login"
            Write-Host "Then:  cloudflared tunnel route dns navine-ai $hostname"
        }
    }
}

$pyBrand = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "Navine AI - Python\configs\brand.yaml"
if (Test-Path $pyBrand) {
    Write-Host ""
    Write-Host "PAI brand config: $pyBrand (public_url should be https://pai.navinecord.dev port 8766)"
}

Write-Host ""
Write-Host "Start stack: powershell -ExecutionPolicy Bypass -File scripts\start_public_stack.ps1"
Write-Host ""
