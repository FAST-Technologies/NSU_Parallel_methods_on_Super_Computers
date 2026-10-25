#!/bin/bash

# Скрипт для сборки и запуска mini-rt с поддержкой OpenMP и замером времени
echo "========================================================"
echo "  Mini-RT OpenMP Build & Run Script"
echo "========================================================"

BUILD_DIR="/home/fast_development/Lab_01/mini-rt-build"
SRC_DIR="/home/fast_development/Lab_01/mini-rt/src"

cd "$BUILD_DIR" || exit 1

echo "[1/4] Cleaning previous build..."
make clean > /dev/null 2>&1

echo "[2/4] Configuring CMake with OpenMP and -O3 optimization..."
cmake "$SRC_DIR" -DCMAKE_CXX_FLAGS="-fopenmp -O3" > /dev/null 2>&1

echo "[3/4] Building project..."
make -j4 > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Ошибка компиляции! Проверьте код."
    exit 1
fi

echo "✅ Build successful!"
echo "--------------------------------------------------------"

echo "[4/4] Running renderer (measuring execution time)..."
cd test || exit 1

echo "Запуск minirt-test..."
# Утилита 'time' покажет реальное (real), пользовательское (user) и системное (sys) время
/usr/bin/time -v ./minirt-test 

echo "--------------------------------------------------------"
echo "Проверка выходных файлов:"
ls -lh *.jpg *.ppm 2>/dev/null || echo "⚠️ Изображение не найдено. Возможно, тесту нужен аргумент (файл сцены)."

echo "========================================================"
echo "Готово! Проверьте папку test на наличие изображения."
