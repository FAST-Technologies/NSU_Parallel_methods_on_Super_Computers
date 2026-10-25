#!/bin/bash

echo "========================================================"
echo "  OpenMP Ray Tracing Benchmark (Anti-Throttling)"
echo "========================================================"

BUILD_DIR="/home/fast_development/Lab_01/mini-rt-build/test"
cd "$BUILD_DIR" || exit 1

# Параметры теста
RES_X=1000
RES_Y=1000
SAMPLES=10
RUNS=3

# Стратегии и количество потоков для теста
STRATEGIES=("dynamic,16" "static" "static,1" "guided,16")
THREADS=(1 2 4 8 12 16 20)

echo "Запуск тестов (это займет несколько минут)..."
echo "Разрешение: ${RES_X}x${RES_Y}, Сэмплов: ${SAMPLES}"
echo "--------------------------------------------------------"

printf "%-12s | %-6s | %-10s\n" "Strategy" "Threads" "Min Time (s)"
printf "%-12s-+-%-6s-+-%-10s\n" "------------" "------" "----------"

for strategy in "${STRATEGIES[@]}"; do
    for threads in "${THREADS[@]}"; do

        min_time=999999.0

        for ((i=1; i<=RUNS; i++)); do
            export OMP_NUM_THREADS=$threads
            export OMP_SCHEDULE="$strategy"

            output=$(./minirt-test $RES_X $RES_Y $SAMPLES 2>&1)
            time_ms=$(echo "$output" | grep "RENDER TIME" | awk '{print $5}')
            if [ -z "$time_ms" ]; then continue; fi
            time_s=$(echo "scale=4; $time_ms / 1000" | bc)

            is_less=$(echo "$time_s < $min_time" | bc)
            if [ "$is_less" -eq 1 ]; then
                min_time=$time_s
            fi
        done
        printf "%-12s | %-6s | %-10s\n" "$strategy" "$threads" "$min_time"
    done
    sleep 3
done

echo "========================================================"
echo "Готово! Скопируйте таблицу выше в Excel для расчета S и E."