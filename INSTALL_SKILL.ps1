param(
    [switch]$InstallDependencies
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $repoRoot "skills\welltestai-analysis"
$dst = Join-Path $env:USERPROFILE ".codex\skills\welltestai-analysis"

if (!(Test-Path $src)) {
    throw "Skill folder not found: $src"
}

New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
Copy-Item -Recurse -Force $src $dst
Write-Host "Installed WellTestAI skill to $dst"
if ($InstallDependencies) {
    python -m pip install numpy pandas scikit-learn joblib PyYAML matplotlib
}
Write-Host "Restart Codex to load the skill."
