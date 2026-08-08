param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$WorkflowArgs
)

$env:PYTHONUTF8 = "1"
& python -X utf8 "C:\Users\23258\.codex\skills\math-modeling-workflow\scripts\workflow.py" @WorkflowArgs --project $PSScriptRoot
exit $LASTEXITCODE
