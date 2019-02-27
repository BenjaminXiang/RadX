#!/usr/bin/pwsh-preview

# It is helper for compilation shaders to SPIR-V

$INDIR="./"
$OUTDIR="../prebuilt/shaders/$VNDR/"
$HRDDIR="../prebuilt/intrusive/$VNDR/"
$RDXO="radix/"

$CMPPROF="-S comp"
$OPTFLAGS="-O --skip-validation --strip-debug --inline-entry-points-exhaustive --strength-reduction --vector-dce --workaround-1209 --replace-invalid-opcode --ccp --unify-const --simplify-instructions --remove-duplicates --combine-access-chains  --convert-local-access-chains --private-to-local --merge-return --merge-blocks --if-conversion --cfg-cleanup --flatten-decorations --freeze-spec-const "

function Pause ($Message = "Press any key to continue . . . ") {
#    if ((Test-Path variable:psISE) -and $psISE) {
#        $Shell = New-Object -ComObject "WScript.Shell"
#        $Button = $Shell.Popup("Click OK to continue.", 0, "Script Paused", 0)
#    }
#    else {     
#        Write-Host -NoNewline $Message
#        [void][System.Console]::ReadKey($true)
#        Write-Host
#    }
}

function Optimize($Name, $Dir = "", $AddArg = "") {
    $ARGS = "$OPTFLAGS $Dir$Name.spv -o $Dir$Name.spv $AddArg"
    $process = start-process -NoNewWindow -Filepath "spirv-opt" -ArgumentList "$ARGS" -PassThru
    #$process.PriorityClass = 'BelowNormal'
    $process.WaitForExit()
    $process.Close()
}

function BuildCompute($Name, $InDir = "", $OutDir = "", $AddArg = "", $AltName = $Name) {
    $ARGS = "$CFLAGSV $CMPPROF $InDir$Name -o $OutDir$AltName.spv $AddArg"
    $process = start-process -NoNewWindow -Filepath "glslangValidator" -ArgumentList "$ARGS" -PassThru
    #$process.PriorityClass = 'BelowNormal'
    $process.WaitForExit()
    $process.Close()
}

function OptimizeMainline($Pfx = "", $RDXI="radix/") {
    # optimize radix sort
    
    Optimize "counting.comp"      "$HRDDIR$RDXI"
    Optimize "partition.comp"     "$HRDDIR$RDXI"
    Optimize "scattering.comp"    "$HRDDIR$RDXI"
    
    Optimize "indiction.comp"     "$HRDDIR$RDXI"
    Optimize "permutation.comp"   "$HRDDIR$RDXI"
}


function BuildAllShaders($Pfx = "", $RDXI="radix/") {
    #[System.Threading.Thread]::CurrentThread.Priority = 'BelowNormal'
    #[System.Threading.Thread]::CurrentThread.Priority = 'Highest'

    new-item -Name $HRDDIR$RDXO -itemtype directory  -Force | Out-Null

    # radix sort
    BuildCompute "counting.comp"     "$INDIR$RDXI" "$HRDDIR$RDXO"
    BuildCompute "partition.comp"    "$INDIR$RDXI" "$HRDDIR$RDXO"
    BuildCompute "scattering.comp"   "$INDIR$RDXI" "$HRDDIR$RDXO"
    
    BuildCompute "indiction.comp"    "$INDIR$RDXI" "$HRDDIR$RDXO"
    BuildCompute "permutation.comp"  "$INDIR$RDXI" "$HRDDIR$RDXO"

    # optimize built shaders
    OptimizeMainline $RDXO
}




