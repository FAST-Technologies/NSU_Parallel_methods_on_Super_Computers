#!/bin/bash

# ==============================================================================
# Настройки путей и параметров
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/build"
EXECUTABLE="$BUILD_DIR/Lab_03"

MATRIX_SIZES=(256 512 1024 2048 4096 8192)
BLOCK_SIZES=(2 4 8 16 32)
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
printf "%-4s | %-5s | %-10s | %-15s | %-15s | %-10s | %-10s | %-10s\n" "N" "Block" "Naive" "Tiled Static" "Tiled Dynamic" "cuBLAS" "Spd (Stat)" "Spd (Dyn)"
echo "------------------------------------------------------------------------------------------------------------------------"

for N in "${MATRIX_SIZES[@]}"; do
    for BS in "${BLOCK_SIZES[@]}"; do

        min_naive=999999.0
        min_tiled_static=999999.0
        min_tiled_dynamic=999999.0
        min_cublas=999999.0

        for i in $(seq 1 $RUNS); do
            # Запускаем программу, подавляем лишний вывод, оставляем только нужные строки
            output=$("$EXECUTABLE" "$N" "$BS" 2>&1)

            # Извлекаем время (формат вывода: "Naive (Global):  X.XXX ms")
            # awk '{print $3}' берет третье слово, которое является числом
            t_naive=$(echo "$output" | grep "Naive (Global):" | awk '{print $3}')
            t_tiled_static=$(echo "$output" | grep "Tiled (Static):" | awk '{print $3}')
            t_tiled_dynamic=$(echo "$output" | grep "Tiled (Dynamic):" | awk '{print $3}')
            t_cublas=$(echo "$output" | grep "cuBLAS (Warm):" | awk '{print $3}')

            # Обновляем минимальные значения (awk отлично справится с одиночными float числами)
            min_naive=$(awk -v t="$t_naive" -v m="$min_naive" 'BEGIN {print (t < m && t != "") ? t : m}')
            min_tiled_static=$(awk -v t="$t_tiled_static" -v m="$min_tiled_static" 'BEGIN {print (t < m && t != "") ? t : m}')
            min_tiled_dynamic=$(awk -v t="$t_tiled_dynamic" -v m="$min_tiled_dynamic" 'BEGIN {print (t < m && t != "") ? t : m}')
            min_cublas=$(awk -v t="$t_cublas" -v m="$min_cublas" 'BEGIN {print (t < m && t != "") ? t : m}')
        done

        # Вычисляем ускорение Tiled относительно Naive
        speedup_static=$(awk -v n="$min_naive" -v t="$min_tiled_static" 'BEGIN {if(t>0) printf "%.2f", n/t; else print "N/A"}')
        speedup_dynamic=$(awk -v n="$min_naive" -v t="$min_tiled_dynamic" 'BEGIN {if(t>0) printf "%.2f", n/t; else print "N/A"}')
        printf "%-4d | %-5d | %-10.3f | %-15.3f | %-15.3f | %-10.3f | %-10.2fx | %-10.2fx\n" \
            "$N" "$BS" "$min_naive" "$min_tiled_static" "$min_tiled_dynamic" "$min_cublas" "$speedup_static" "$speedup_dynamic"
    done
done

echo "----------------------------------------------------------------------------------------"
echo "✅ Бенчмарк успешно завершен!"
echo "💡 Совет: Скопируйте эту таблицу напрямую в ваш отчет (Assignment 03)."