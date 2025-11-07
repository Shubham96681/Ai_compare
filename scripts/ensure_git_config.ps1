# PowerShell script to ensure git user email is set correctly
# Run this before committing to ensure correct git user email

$desiredEmail = "infofitsoftware@gmail.com"
$currentEmail = git config user.email

if ($currentEmail -ne $desiredEmail) {
    Write-Host "⚠️  Git email is not set correctly. Current: $currentEmail" -ForegroundColor Yellow
    Write-Host "🔧 Setting git email to: $desiredEmail" -ForegroundColor Green
    git config user.email $desiredEmail
    git config user.name "InfoFit Software"
    Write-Host "✅ Git configuration updated!" -ForegroundColor Green
} else {
    Write-Host "✅ Git email is correctly set: $currentEmail" -ForegroundColor Green
}

