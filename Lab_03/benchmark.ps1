$ErrorActionPreference = "Stop"
$BUILD_DIR = Join-Path $PSScriptRoot "build"
$EXECUTABLE = Join-Path $BUILD_DIR "Lab_03.exe"
$MATRIX_SIZES = @(256, 512, 1024, 2048, 4096, 8192)
$BLOCK_SIZES = @(2, 4, 8, 16, 32)
$RUNS = 3

Write-Host "========================================================"
Write-Host "  Бенчмарк умножения матриц (CUDA / cuBLAS)"
Write-Host "  Тестируемые размеры матриц (N): $($MATRIX_SIZES -join ', ')"
Write-Host "  Тестируемые размеры блоков: $($BLOCK_SIZES -join ', ')"
Write-Host "  Количество прогонов на конфигурацию: $RUNS"
Write-Host "========================================================"

if (-Not (Test-Path $EXECUTABLE)) {
    Write-Host "Ошибка: Исполняемый файл не найден по пути: $EXECUTABLE" -ForegroundColor Red
    exit 1
}

Write-Host "Запуск тестов (выбирается минимальное время из $RUNS запусков)..."
Write-Host "----------------------------------------------------------------------------------------"
$format = "{0,-4} | {1,-5} | {2,-10} | {3,-15} | {4,-15} | {5,-10} | {6,-10} | {7,-10}"
Write-Host ($format -f "N", "Block", "Naive", "Tiled Static", "Tiled Dynamic", "cuBLAS", "Spd (Stat)", "Spd (Dyn)")
Write-Host "------------------------------------------------------------------------------------------------------------------------"

foreach ($N in $MATRIX_SIZES) {
    foreach ($BS in $BLOCK_SIZES) {
        $min_naive = [double]::MaxValue
        $min_tiled_static = [double]::MaxValue
        $min_tiled_dynamic = [double]::MaxValue
        $min_cublas = [double]::MaxValue

        for ($i = 1; $i -le $RUNS; $i++) {
            $output = & $EXECUTABLE $N $BS 2>&1 | Out-String
            $match_naive = [regex]::Match($output, "Naive \(Global\):\s+([\d\.]+)")
            $match_static = [regex]::Match($output, "Tiled \(Static\):\s+([\d\.]+)")
            $match_dynamic = [regex]::Match($output, "Tiled \(Dynamic\):\s+([\d\.]+)")
            $match_cublas = [regex]::Match($output, "cuBLAS \(Warm\):\s+([\d\.]+)")

            if ($match_naive.Success) { $t_naive = [double]$match_naive.Groups[1].Value } else { $t_naive = 0 }
            if ($match_static.Success) { $t_static = [double]$match_static.Groups[1].Value } else { $t_static = 0 }
            if ($match_dynamic.Success) { $t_dynamic = [double]$match_dynamic.Groups[1].Value } else { $t_dynamic = 0 }
            if ($match_cublas.Success) { $t_cublas = [double]$match_cublas.Groups[1].Value } else { $t_cublas = 0 }

            if ($t_naive -gt 0 -and $t_naive -lt $min_naive) { $min_naive = $t_naive }
            if ($t_static -gt 0 -and $t_static -lt $min_tiled_static) { $min_tiled_static = $t_static }
            if ($t_dynamic -gt 0 -and $t_dynamic -lt $min_tiled_dynamic) { $min_tiled_dynamic = $t_dynamic }
            if ($t_cublas -gt 0 -and $t_cublas -lt $min_cublas) { $min_cublas = $t_cublas }
        }

        if ($min_tiled_static -gt 0) { $speedup_static = $min_naive / $min_tiled_static } else { $speedup_static = 0 }
        if ($min_tiled_dynamic -gt 0) { $speedup_dynamic = $min_naive / $min_tiled_dynamic } else { $speedup_dynamic = 0 }

        $str_stat = "{0:N2} x" -f $speedup_static
        $str_dyn = "{0:N2} x" -f $speedup_dynamic
        
        Write-Host ($format -f $N, $BS, $min_naive.ToString("0.000"), $min_tiled_static.ToString("0.000"), $min_tiled_dynamic.ToString("0.000"), $min_cublas.ToString("0.000"), $str_stat, $str_dyn)
    }
}

Write-Host "----------------------------------------------------------------------------------------"
Write-Host "Бенчмарк успешно завершен!" -ForegroundColor Green
