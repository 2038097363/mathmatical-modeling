[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('研究生', '本科生', '专科生')]
    [string]$Group,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^CM[0-9]{7}$')]
    [string]$CompetitionId,

    [Parameter(Mandatory = $true)]
    [string]$OutputDocx,

    [string]$MetadataFile = '',
    [string]$FigureMap = '',
    [string]$ContentMap = '',
    [string]$RenderDir = ''
)

$ErrorActionPreference = 'Stop'
$scriptDir = $PSScriptRoot
$projectRoot = (Resolve-Path (Join-Path $scriptDir '..\..')).Path
if (-not $MetadataFile) { $MetadataFile = Join-Path $scriptDir 'paper_metadata.yml' }
if (-not $FigureMap) { $FigureMap = Join-Path $scriptDir 'figure_map.json' }
if (-not $ContentMap) { $ContentMap = Join-Path $scriptDir 'content_map.json' }
$metadataPath = (Resolve-Path -LiteralPath $MetadataFile).Path
$figureMapPath = (Resolve-Path -LiteralPath $FigureMap).Path
$contentMapPath = (Resolve-Path -LiteralPath $ContentMap).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputDocx)
$expectedStem = 'A' + $CompetitionId
if ([System.IO.Path]::GetFileNameWithoutExtension($outputPath) -cne $expectedStem) {
    throw "Formal DOCX filename must be $expectedStem.docx"
}
if ([System.IO.Path]::GetExtension($outputPath).ToLowerInvariant() -ne '.docx') {
    throw 'OutputDocx must use the .docx extension'
}
if (Test-Path -LiteralPath $outputPath) { throw "OutputDocx already exists: $outputPath" }
$outputParent = Split-Path -Parent $outputPath
if (-not (Test-Path -LiteralPath $outputParent)) { New-Item -ItemType Directory -Path $outputParent -Force | Out-Null }
if (-not $RenderDir) { $RenderDir = Join-Path $outputParent ($expectedStem + '-render') }
$renderPath = [System.IO.Path]::GetFullPath($RenderDir)
if (Test-Path -LiteralPath $renderPath) { throw "RenderDir must not already exist: $renderPath" }

$python = (Get-Command python -ErrorAction Stop).Source
$readinessPath = Join-Path $outputParent ($expectedStem + '.readiness.json')
$buildManifest = Join-Path $outputParent ($expectedStem + '.build.json')
$structureAudit = Join-Path $outputParent ($expectedStem + '.structure-audit.json')

& $python (Join-Path $scriptDir 'check_docx_readiness.py') `
    --metadata-file $metadataPath `
    --figure-map $figureMapPath `
    --content-map $contentMapPath `
    --json $readinessPath
if ($LASTEXITCODE -ne 0) { throw "Formal readiness gate failed; see $readinessPath" }

& $python (Join-Path $scriptDir 'build_docx.py') `
    --group $Group `
    --competition-id $CompetitionId `
    --metadata-file $metadataPath `
    --figure-map $figureMapPath `
    --content-map $contentMapPath `
    --manifest $buildManifest `
    --output $outputPath
if ($LASTEXITCODE -ne 0) { throw 'Formal DOCX build failed' }

& $python (Join-Path $scriptDir 'audit_docx.py') $outputPath --json $structureAudit
if ($LASTEXITCODE -ne 0) { throw "Formal structure/privacy audit failed; see $structureAudit" }

& (Join-Path $scriptDir 'render_docx.ps1') `
    -InputDocx $outputPath `
    -OutputDir $renderPath `
    -Engine Auto `
    -Dpi 150 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Formal WPS/LibreOffice + Poppler render failed' }

$pdfPath = Join-Path $renderPath ($expectedStem + '.pdf')
if (-not (Test-Path -LiteralPath $pdfPath)) { throw "Expected rendered PDF is missing: $pdfPath" }
[ordered]@{
    docx = $outputPath
    pdf = $pdfPath
    readiness = $readinessPath
    buildManifest = $buildManifest
    structureAudit = $structureAudit
    renderAudit = (Join-Path $renderPath 'render-audit.json')
    visualReviewRequired = $true
} | ConvertTo-Json -Depth 5
