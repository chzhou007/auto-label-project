param(
  [string]$Config = "configs/autolabel.yaml",
  [string]$MetadataDir = "data/processed/metadata",
  [string]$AssetBaseDir = "",
  [string]$OutputDir = "data/qc",
  [int]$Limit = 20,
  [double]$SamplingRatio = 0.0
)

$ErrorActionPreference = "Stop"

if (-not $env:QWEN397B_API_KEY) {
  throw "QWEN397B_API_KEY is not set. Set it in the current PowerShell session before running VLM QC."
}

if (-not $env:QWEN_GEOMETRY_MODEL) {
  $env:QWEN_GEOMETRY_MODEL = "aios-smart-eye-vlm"
}

$argsList = @(
  "scripts/run_qc_agent.py",
  "--config", $Config,
  "--metadata-dir", $MetadataDir,
  "--output-dir", $OutputDir,
  "--sampling-ratio", "$SamplingRatio",
  "--enable-vlm",
  "--limit", "$Limit"
)

if ($AssetBaseDir) {
  $argsList += @("--asset-base-dir", $AssetBaseDir)
}

python @argsList
