$ErrorActionPreference = "Stop"

$BUILD_DIR = "$PSScriptRoot\build"
$EXECUTABLE = "$BUILD_DIR\Lab_03.exe"

$MATRIX_SIZES = @(256, 512, 1024, 2048, 4096, 8192)
$BLOCK_SIZES = @(2, 4, 8, 16, 32)
$RUNS = 3

Write-Host "========================================================"
Write-Host "  Бенчмарк умножения матриц (CUDA / cuBLAS)"
Write-Host "  Тестируемые размеры матриц (N): $($MATRIX_SIZES -join ' ')"
Write-Host "  Тестируемые размеры блоков: $($BLOCK_SIZES -join ' ')"
Write-Host "  Количество прогонов на конфигурацию: $RUNS"
Write-Host "========================================================"

if (-Not (Test-Path $EXECUTABLE)) {
    Write-Host "❌ Ошибка: Исполняемый файл не найден по пути: $EXECUTABLE" -ForegroundColor Red
    Write-Host "💡 Сначала соберите проект, выполнив:"
    Write-Host "   cmake --build $BUILD_DIR --config Release"
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
            # Запускаем программу и получаем весь вывод как строку
            $output = & $EXECUTABLE $N $BS 2>&1 | Out-String
            
            # Парсим значения с помощью регулярных выражений
            $t_naive = [double]($output -split "`n" | Where-Object { $_ -match "Naive \(Global\):\s+([\d\.]+)" } | ForEach-Object { $matches[1] })
            $t_tiled_static = [double]($output -split "`n" | Where-Object { $_ -match "Tiled \(Static\):\s+([\d\.]+)" } | ForEach-Object { $matches[1] })
            $t_tiled_dynamic = [double]($output -split "`n" | Where-Object { $_ -match "Tiled \(Dynamic\):\s+([\d\.]+)" } | ForEach-Object { $matches[1] })
            $t_cublas = [double]($output -split "`n" | Where-Object { $_ -match "cuBLAS \(Warm\):\s+([\d\.]+)" } | ForEach-Object { $matches[1] })

            # Обновляем минимальные значения
            if ($t_naive -gt 0 -and $t_naive -lt $min_naive) { $min_naive = $t_naive }
            if ($t_tiled_static -gt 0 -and $t_tiled_static -lt $min_tiled_static) { $min_tiled_static = $t_tiled_static }
            if ($t_tiled_dynamic -gt 0 -and $t_tiled_dynamic -lt $min_tiled_dynamic) { $min_tiled_dynamic = $t_tiled_dynamic }
            if ($t_cublas -gt 0 -and $t_cublas -lt $min_cublas) { $min_cublas = $t_cublas }
        }

        # Вычисляем ускорение
        $speedup_static = if ($min_tiled_static -gt 0) { $min_naive / $min_tiled_static } else { 0 }
        $speedup_dynamic = if ($min_tiled_dynamic -gt 0) { $min_naive / $min_tiled_dynamic } else { 0 }

        # Выводим строку таблицы
        Write-Host ($format -f $N, $BS, $min_naive.ToString("0.000"), $min_tiled_static.ToString("0.000"), $min_tiled_dynamic.ToString("0.000"), $min_cublas.ToString("0.000"), "$($speedup_static.ToString("0.00")) x", "$($speedup_dynamic.ToString("0.00")) x")
    }
}

Write-Host "----------------------------------------------------------------------------------------"
Write-Host "Бенчмарк успешно завершен!" -ForegroundColor Green
