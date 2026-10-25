/**
 * @file main.cpp
 * @brief Лабораторная работа №1, п. 2.2: Исследование загрузки ядер при распараллеливании циклов.
 * @details Программа демонстрирует разницу в загрузке процессорных ядер при выполнении
 *          вычислительно сложной задачи с использованием OpenMP и без него.
 */

#include <iostream>
#include <chrono>
#include <omp.h>
#include <cmath>

/**
 * @brief Чистая функция, выполняющая ресурсоемкие вычисления.
 * @details Функция является "чистой" (pure), так как:
 *          1. Её возвращаемое значение зависит только от входного аргумента.
 *          2. Она не имеет побочных эффектов (не изменяет глобальные переменные, не выполняет I/O).
 *          Цикл подобран так, чтобы выполнение занимало заметное время (порядка миллисекунд на итерацию).
 * @param x Входное значение типа double.
 * @return Результат последовательного применения тригонометрических функций.
 */
double heavy_function(double x) {
    double res = x;
    // Имитация сложных вычислений, загружающих арифметико-логическое устройство (ALU)
    for (int i = 0; i < 50000000; ++i) {
        res = std::sin(res) * std::cos(res);
    }
    return res;
}

int main() {
    const int N = 10;
    double a[N], b[N];
    for (int i = 0; i < N; i++)
        b[i] = static_cast<double>(i);

    std::cout << "Starting computation with N = " << N << "...\n";

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        a[i] = heavy_function(b[i]);
        b[i] = 2 * a[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation finished.\n";
    std::cout << "Elapsed time: " << diff.count() << " seconds\n";
    std::cout << "Check value a[0]: " << a[0] << "\n";
    return 0;
}
