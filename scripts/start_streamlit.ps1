param(
    [int]$Port = 8501,
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot "venv\Scripts\python.exe"
$appPath = Join-Path $repoRoot "streamlit_app\app.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment Python not found at $pythonExe. Create and activate venv first."
    exit 1
}

if (-not (Test-Path $appPath)) {
    Write-Error "Streamlit entry file not found at $appPath."
    exit 1
}

if (-not $KeepExisting) {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listeners) {
        $processIds = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($processId in $processIds) {
            try {
                Stop-Process -Id $processId -Force -ErrorAction Stop
                Write-Host "Stopped process $processId on port $Port"
            }
            catch {
                Write-Warning ("Could not stop process {0} on port {1}: {2}" -f $processId, $Port, $_.Exception.Message)
            }
        }
    }
}

Write-Host "Starting Streamlit on http://localhost:$Port"
& $pythonExe -m streamlit run $appPath --server.port $Port --server.headless true --server.runOnSave false
