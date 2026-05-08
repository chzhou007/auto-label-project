param(
    [string]$Config = "configs/autolabel.yaml",
    [string]$BatchName = "",
    [string]$RunRoot = "",
    [string]$RawImages = "",
    [string]$RawVideos = "",
    [Nullable[int]]$VideoFrameStride = $null,
    [Nullable[int]]$VideoMaxFrames = $null,
    [ValidateSet("cpu", "gpu", "auto")]
    [string]$VideoDecodeMode = "",
    [string]$FfmpegPath = "",
    [string]$VideoGpuHwaccel = "",
    [switch]$NoVideoGpuFallback,
    [ValidateSet("fail", "skip")]
    [string]$VideoErrorPolicy = "",
    [switch]$DryRunModels,
    [switch]$DryRunGeometry,
    [switch]$DryRunClassification,
    [switch]$NoClassify,
    [switch]$InstallDeps,
    [switch]$UpdateExportStatus
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($BatchName)) {
    $BatchName = "batch_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path "data/runs" $BatchName
}

$ImageSequenceDir = Join-Path $RunRoot "image_sequence"
$ManifestPath = Join-Path $RunRoot "manifest.csv"
$ProcessedRoot = Join-Path $RunRoot "processed"
$MetadataDir = Join-Path $ProcessedRoot "metadata"
$ExportDir = Join-Path $RunRoot "exports"
$LabelStudioOutput = Join-Path $ExportDir "labelstudio_import.json"
$LabelStudioConfig = Join-Path $ExportDir "labelstudio_config.xml"

Write-Host "== AutoLabel batch =="
Write-Host "Project root:      $ProjectRoot"
Write-Host "Config:            $Config"
Write-Host "Batch name:        $BatchName"
Write-Host "Run root:          $RunRoot"
Write-Host ""

if ($InstallDeps) {
    Write-Host "== Installing dependencies =="
    python -m pip install -r requirements.txt
    Write-Host ""
}

Write-Host "== Step 1/4: preprocess raw images/videos =="
$PreprocessArgs = @(
    "scripts/run_preprocess.py",
    "--config", $Config,
    "--output", $ImageSequenceDir,
    "--manifest", $ManifestPath
)

if (-not [string]::IsNullOrWhiteSpace($RawImages)) {
    $PreprocessArgs += @("--raw-images", $RawImages)
}
if (-not [string]::IsNullOrWhiteSpace($RawVideos)) {
    $PreprocessArgs += @("--raw-videos", $RawVideos)
}
if ($null -ne $VideoFrameStride) {
    $PreprocessArgs += @("--video-frame-stride", "$VideoFrameStride")
}
if ($null -ne $VideoMaxFrames) {
    $PreprocessArgs += @("--video-max-frames", "$VideoMaxFrames")
}
if (-not [string]::IsNullOrWhiteSpace($VideoDecodeMode)) {
    $PreprocessArgs += @("--video-decode-mode", $VideoDecodeMode)
}
if (-not [string]::IsNullOrWhiteSpace($FfmpegPath)) {
    $PreprocessArgs += @("--ffmpeg-path", $FfmpegPath)
}
if (-not [string]::IsNullOrWhiteSpace($VideoGpuHwaccel)) {
    $PreprocessArgs += @("--video-gpu-hwaccel", $VideoGpuHwaccel)
}
if ($NoVideoGpuFallback) {
    $PreprocessArgs += "--no-video-gpu-fallback"
}
if (-not [string]::IsNullOrWhiteSpace($VideoErrorPolicy)) {
    $PreprocessArgs += @("--video-error-policy", $VideoErrorPolicy)
}

python @PreprocessArgs
Write-Host ""

Write-Host "== Step 2/4: detect person boxes and classify crops =="
$PipelineArgs = @(
    "scripts/run_pipeline.py",
    "--config", $Config,
    "--branches", "direct",
    "--manifest", $ManifestPath,
    "--processed-root", $ProcessedRoot
)

if ($DryRunModels) {
    $PipelineArgs += "--dry-run-models"
}
if ($DryRunGeometry) {
    $PipelineArgs += "--dry-run-geometry"
}
if ($DryRunClassification) {
    $PipelineArgs += "--dry-run-classification"
}
if ($NoClassify) {
    $PipelineArgs += "--no-classify"
}

python @PipelineArgs
Write-Host ""

Write-Host "== Step 3/4: validate AutoLabelSample metadata =="
python scripts/validate_sample.py --metadata-dir $MetadataDir
Write-Host ""

Write-Host "== Step 4/4: export Label Studio import files =="
$ExportArgs = @(
    "scripts/export_labelstudio.py",
    "--config", $Config,
    "--metadata-dir", $MetadataDir,
    "--output", $LabelStudioOutput
)
if ($UpdateExportStatus) {
    $ExportArgs += "--update-samples"
}

python @ExportArgs
Write-Host ""

$MetadataCount = @(Get-ChildItem -LiteralPath $MetadataDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
$CropDir = Join-Path $ProcessedRoot "crops"
$CropCount = @(Get-ChildItem -LiteralPath $CropDir -Filter "*.jpg" -File -ErrorAction SilentlyContinue).Count

Write-Host "== Done =="
Write-Host "Metadata count:    $MetadataCount"
Write-Host "Crop count:        $CropCount"
Write-Host "Manifest:          $ManifestPath"
Write-Host "Metadata dir:      $MetadataDir"
Write-Host "Crop dir:          $CropDir"
Write-Host "Label Studio JSON: $LabelStudioOutput"
Write-Host "Label Studio XML:  $LabelStudioConfig"
Write-Host ""
Write-Host "For no-key validation, use: -DryRunModels"
