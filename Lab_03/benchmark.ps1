
$ErrorActionPreference = "Stop"

$RESULTS_DIR = Join-Path $PSScriptRoot "results"

$BUILDS = @(
    @{
        Name = "float"
        Directory = "build-float"
    },
    @{
        Name = "double"
        Directory = "build-double"
    }
)

Write-Host "========================================================"
Write-Host "  CUDA Matrix Multiplication Benchmark"
Write-Host "  Precision: float + double"
Write-Host "  Matrix sizes are taken from config.hpp (MATRIX_SIZES)"
Write-Host "  Repeats and warmup are handled by the C++ program"
Write-Host "========================================================"

New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null

foreach ($BUILD in $BUILDS) {
    $BUILD_DIR = Join-Path $PSScriptRoot $BUILD.Directory
    $EXECUTABLE = Join-Path $BUILD_DIR "Lab_03.exe"
    $CSV_SOURCE = Join-Path $BUILD_DIR "cuda_lab3_results.csv"
    $CSV_DESTINATION = Join-Path $RESULTS_DIR "cuda_lab3_$($BUILD.Name)_results.csv"

    Write-Host ""
    Write-Host "--------------------------------------------------------"
    Write-Host "Running $($BUILD.Name) benchmark"
    Write-Host "Executable: $EXECUTABLE"
    Write-Host "--------------------------------------------------------"

    if (-not (Test-Path $EXECUTABLE)) {
        throw "Executable not found: $EXECUTABLE. Build the $($BUILD.Name) configuration first."
    }

    Push-Location $BUILD_DIR
    try {
        & ".\Lab_03.exe"

        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark ($($BUILD.Name)) failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path $CSV_SOURCE)) {
        throw "Results CSV was not created: $CSV_SOURCE"
    }

    Copy-Item -Force $CSV_SOURCE $CSV_DESTINATION

    Write-Host "Saved: $CSV_DESTINATION" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================================"
Write-Host "All benchmarks completed successfully."
Write-Host "Results directory: $RESULTS_DIR"
Write-Host "========================================================"
