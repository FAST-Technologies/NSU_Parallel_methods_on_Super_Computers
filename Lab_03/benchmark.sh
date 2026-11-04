#!/bin/bash

# ==============================================================================
# Настройки путей и параметров
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/build"
EXECUTABLE="$BUILD_DIR/Lab_03"

MATRIX_SIZES=(256 512 1024 2048)
BLOCK_SIZES=(16 32)
RUNS=3

echo "========================================================"
echo "  Бенчмарк умножения матриц (CUDA / cuBLAS)"
echo "  Тестируемые размеры матриц (N): ${MATRIX_SIZES[*]}"
echo "  Тестируемые размеры блоков: ${BLOCK_SIZES[*]}"
echo "  Количество прогонов на конфигурацию: $RUNS"
echo "========================================================"

if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ Ошибка: Исполняемый файл не найден по пути: $EXECUTABLE"
    echo "💡 Сначала соберите проект, выполнив:"
    echo "   cmake --build $BUILD_DIR --config Release"
    exit 1
fi

echo "Запуск тестов (выбирается минимальное время из $RUNS запусков)..."
echo "----------------------------------------------------------------------------------------"
printf "%-6s | %-8s | %-12s | %-12s | %-12s | %-10s\n" "N" "Block" "Naive (ms)" "Tiled (ms)" "cuBLAS (ms)" "Speedup"
echo "----------------------------------------------------------------------------------------"

for N in "${MATRIX_SIZES[@]}"; do
    for BS in "${BLOCK_SIZES[@]}"; do

        min_naive=999999.0
        min_tiled=999999.0
        min_cublas=999999.0

        for i in $(seq 1 $RUNS); do
            # Запускаем программу, подавляем лишний вывод, оставляем только нужные строки
            output=$("$EXECUTABLE" "$N" "$BS" 2>&1)

            # Извлекаем время (формат вывода: "Naive (Global):  X.XXX ms")
            # awk '{print $3}' берет третье слово, которое является числом
            t_naive=$(echo "$output" | grep "Naive" | awk '{print $3}')
            t_tiled=$(echo "$output" | grep "Tiled" | awk '{print $3}')
            t_cublas=$(echo "$output" | grep "cuBLAS" | awk '{print $3}')

            # Обновляем минимальные значения с помощью awk (сравнение float)
            min_naive=$(awk -v t="$t_naive" -v m="$min_naive" 'BEGIN {print (t < m && t != "") ? t : m}')
            min_tiled=$(awk -v t="$t_tiled" -v m="$min_tiled" 'BEGIN {print (t < m && t != "") ? t : m}')
            min_cublas=$(awk -v t="$t_cublas" -v m="$min_cublas" 'BEGIN {print (t < m && t != "") ? t : m}')
        done

        # Вычисляем ускорение Tiled относительно Naive
        speedup=$(awk -v n="$min_naive" -v t="$min_tiled" 'BEGIN {if(t>0) printf "%.2f", n/t; else print "N/A"}')
        printf "%-6d | %-8d | %-12.3f | %-12.3f | %-12.3f | %-10.2fx\n" "$N" "$BS" "$min_naive" "$min_tiled" "$min_cublas" "$speedup"

    done
done

echo "----------------------------------------------------------------------------------------"
echo "✅ Бенчмарк успешно завершен!"
echo "💡 Совет: Скопируйте эту таблицу напрямую в ваш отчет (Assignment 03)."