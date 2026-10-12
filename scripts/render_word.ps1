param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open([System.IO.Path]::GetFullPath($InputPath), $false, $true)
    $document.Repaginate()
    $document.ExportAsFixedFormat([System.IO.Path]::GetFullPath($OutputPath), 17)
} finally {
    if ($document) { $document.Close(0) }
    if ($word) { $word.Quit() }
    if ($document) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($word) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
}

