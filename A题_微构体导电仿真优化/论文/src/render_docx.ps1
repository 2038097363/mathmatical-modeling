[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [ValidateSet('Auto', 'WPS', 'Word', 'LibreOffice')]
    [string]$Engine = 'Auto',

    [ValidateRange(96, 600)]
    [int]$Dpi = 150
)

$ErrorActionPreference = 'Stop'

function Resolve-Executable([string[]]$Names, [string[]]$KnownPaths) {
    foreach ($name in $Names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
    }
    foreach ($path in $KnownPaths) {
        if (Test-Path -LiteralPath $path) { return (Resolve-Path -LiteralPath $path).Path }
    }
    return $null
}

function Export-With-Com([string]$ProgId, [string]$DocxPath, [string]$PdfPath) {
    $app = $null
    $doc = $null
    try {
        $app = New-Object -ComObject $ProgId
        $app.Visible = $false
        try { $app.DisplayAlerts = 0 } catch {}
        $doc = $app.Documents.Open($DocxPath, $false, $true)
        try { [void]$doc.Fields.Update() } catch {}
        try {
            foreach ($section in $doc.Sections) {
                foreach ($kind in 1, 2, 3) {
                    try { [void]$section.Footers.Item($kind).Range.Fields.Update() } catch {}
                }
            }
        } catch {}
        try { [void]$doc.Repaginate() } catch {}
        $doc.ExportAsFixedFormat($PdfPath, 17)
        if (-not (Test-Path -LiteralPath $PdfPath) -or (Get-Item -LiteralPath $PdfPath).Length -eq 0) {
            throw "$ProgId did not create a non-empty PDF"
        }
        $appVersion = ''
        $appPath = ''
        $pageEstimate = $null
        try { $appVersion = [string]$app.Version } catch {}
        try { $appPath = [string]$app.Path } catch {}
        try { $pageEstimate = [int]$doc.ComputeStatistics(2) } catch {}
        return [ordered]@{
            engine = $ProgId
            version = $appVersion
            applicationPath = $appPath
            pages = $pageEstimate
        }
    } finally {
        if ($doc) {
            try { $doc.Close(0) } catch {}
            try { [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc) } catch {}
        }
        if ($app) {
            try { $app.Quit() } catch {}
            try { [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($app) } catch {}
        }
        [gc]::Collect()
        [gc]::WaitForPendingFinalizers()
    }
}

function Export-With-LibreOffice([string]$DocxPath, [string]$PdfPath, [string]$OutputPath) {
    $soffice = Resolve-Executable @('soffice', 'libreoffice') @(
        'C:\Program Files\LibreOffice\program\soffice.exe',
        'C:\Program Files (x86)\LibreOffice\program\soffice.exe'
    )
    if (-not $soffice) { throw 'LibreOffice soffice executable was not found' }
    $profile = Join-Path ([System.IO.Path]::GetTempPath()) ('huashu-docx-render-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $profile | Out-Null
    try {
        $profileUri = ([uri]$profile).AbsoluteUri
        $profileArgument = '-env:UserInstallation=' + $profileUri
        & $soffice $profileArgument '--headless' '--norestore' '--convert-to' 'pdf' '--outdir' $OutputPath $DocxPath
        if ($LASTEXITCODE -ne 0) { throw "LibreOffice exited with code $LASTEXITCODE" }
        $emitted = Join-Path $OutputPath (([System.IO.Path]::GetFileNameWithoutExtension($DocxPath)) + '.pdf')
        if (-not (Test-Path -LiteralPath $emitted)) { throw 'LibreOffice did not emit the expected PDF' }
        if ($emitted -ne $PdfPath) { Move-Item -LiteralPath $emitted -Destination $PdfPath }
        return [ordered]@{ engine = 'LibreOffice'; version = ''; applicationPath = $soffice; pages = $null }
    } finally {
        if (Test-Path -LiteralPath $profile) {
            $resolvedProfile = (Resolve-Path -LiteralPath $profile).Path
            $resolvedTemp = (Resolve-Path -LiteralPath ([System.IO.Path]::GetTempPath())).Path
            if ($resolvedProfile.StartsWith($resolvedTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
                Remove-Item -LiteralPath $resolvedProfile -Recurse -Force
            }
        }
    }
}

$resolvedInput = (Resolve-Path -LiteralPath $InputDocx).Path
if ([System.IO.Path]::GetExtension($resolvedInput).ToLowerInvariant() -ne '.docx') {
    throw 'InputDocx must be a .docx file'
}

$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path -LiteralPath $outputPath)) {
    New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
}
$resolvedOutput = (Resolve-Path -LiteralPath $outputPath).Path
if (Get-ChildItem -LiteralPath $resolvedOutput -Force | Select-Object -First 1) {
    throw 'OutputDir must be empty; use a new directory for every render iteration'
}

$stem = [System.IO.Path]::GetFileNameWithoutExtension($resolvedInput)
$pdfPath = Join-Path $resolvedOutput ($stem + '.pdf')
$attempts = @()
$export = $null

$engines = if ($Engine -eq 'Auto') { @('WPS', 'Word', 'LibreOffice') } else { @($Engine) }
foreach ($candidate in $engines) {
    try {
        if ($candidate -eq 'WPS') {
            $lastError = $null
            foreach ($progId in @('KWPS.Application', 'wps.Application')) {
                try {
                    $export = Export-With-Com $progId $resolvedInput $pdfPath
                    break
                } catch {
                    $lastError = $_.Exception.Message
                }
            }
            if (-not $export) { throw "WPS COM export failed: $lastError" }
        } elseif ($candidate -eq 'Word') {
            $export = Export-With-Com 'Word.Application' $resolvedInput $pdfPath
        } else {
            $export = Export-With-LibreOffice $resolvedInput $pdfPath $resolvedOutput
        }
        break
    } catch {
        $attempts += [ordered]@{ engine = $candidate; error = $_.Exception.Message }
        $export = $null
    }
}
if (-not $export) {
    throw ('All render engines failed: ' + ($attempts | ConvertTo-Json -Compress -Depth 5))
}

$pdfInfo = Resolve-Executable @('pdfinfo') @('C:\texlive\2026\bin\windows\pdfinfo.exe')
$pdfToPpm = Resolve-Executable @('pdftoppm') @('C:\texlive\2026\bin\windows\pdftoppm.exe')
if (-not $pdfInfo -or -not $pdfToPpm) { throw 'Poppler pdfinfo and pdftoppm are required' }

$infoLines = & $pdfInfo $pdfPath 2>&1
if ($LASTEXITCODE -ne 0) { throw "pdfinfo failed with code $LASTEXITCODE" }
$pageLine = $infoLines | Where-Object { $_ -match '^Pages:\s+(\d+)' } | Select-Object -First 1
if (-not $pageLine) { throw 'pdfinfo did not report a page count' }
[int]$pageCount = [regex]::Match([string]$pageLine, '^Pages:\s+(\d+)').Groups[1].Value
$pdfMetadata = [ordered]@{}
foreach ($field in @('Title', 'Subject', 'Keywords', 'Author')) {
    $line = $infoLines | Where-Object { $_ -match ('^' + $field + ':') } | Select-Object -First 1
    $value = if ($line) { ([string]$line).Substring(([string]$line).IndexOf(':') + 1).Trim() } else { '' }
    $pdfMetadata[$field] = $value
}
$nonAnonymous = @($pdfMetadata.GetEnumerator() | Where-Object { $_.Value })
if ($nonAnonymous.Count -gt 0) {
    throw ('PDF metadata must be anonymous: ' + ($nonAnonymous | ConvertTo-Json -Compress))
}

$prefix = Join-Path $resolvedOutput 'page'
& $pdfToPpm '-png' '-r' $Dpi $pdfPath $prefix
if ($LASTEXITCODE -ne 0) { throw "pdftoppm failed with code $LASTEXITCODE" }
$pngs = @(Get-ChildItem -LiteralPath $resolvedOutput -Filter 'page-*.png' | Sort-Object {
    [int]([regex]::Match($_.BaseName, '(\d+)$').Groups[1].Value)
})
if ($pngs.Count -ne $pageCount) {
    throw "Rendered PNG count $($pngs.Count) does not match PDF page count $pageCount"
}

$audit = [ordered]@{
    schemaVersion = '1.0'
    inputDocx = $resolvedInput
    inputSha256 = (Get-FileHash -LiteralPath $resolvedInput -Algorithm SHA256).Hash
    outputDirectory = $resolvedOutput
    pdf = $pdfPath
    pdfSha256 = (Get-FileHash -LiteralPath $pdfPath -Algorithm SHA256).Hash
    pageCount = $pageCount
    dpi = $Dpi
    pdfMetadata = $pdfMetadata
    engine = $export
    failedAttempts = $attempts
    pngs = @($pngs | ForEach-Object { $_.FullName })
    visualReviewRequired = $true
}
$auditPath = Join-Path $resolvedOutput 'render-audit.json'
$audit | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $auditPath -Encoding utf8
$audit | ConvertTo-Json -Depth 10
