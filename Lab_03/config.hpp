
/**
 * @file config.hpp
 * @brief Общая конфигурация лабораторной работы №3.
 *
 * @details Содержит настройки точности вычислений, размеров CUDA-блоков
 *          и параметров проведения бенчмаркинга.
 *
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 15.11.2026.
 * @version 2.0.
 */

#pragma once

#include <cstddef>

/**
 * @brief Базовый тип элементов матриц.
 *
 * @details Тип определяется параметром MATRIX_USE_DOUBLE,
 *          передаваемым при конфигурации проекта через CMake:
 *          - MATRIX_USE_DOUBLE=0 — float;
 *          - MATRIX_USE_DOUBLE=1 — double.
 *
 *          Один и тот же тип используется CPU- и GPU-реализациями
 *          умножения матриц.
 */
#if defined(MATRIX_USE_DOUBLE) && MATRIX_USE_DOUBLE
using BASE_TYPE = double;
#else
using BASE_TYPE = float;
#endif

/**
 * @brief Размер стороны тайла CUDA-блока.
 *
 * @details Определяет размеры двумерного блока потоков TILE_SIZE x TILE_SIZE
 *          и соответствующих тайлов shared memory.
 *
 * @note Для проведения экспериментов можно использовать значения 16 или 32,
 *       если они поддерживаются выбранной реализацией CUDA-ядер.
 */
inline constexpr int TILE_SIZE = 16;

/**
 * @brief Количество CUDA streams в потоковом бенчмарке.
 */
inline constexpr int NUM_STREAMS = 4;

/**
 * @brief Количество повторных измерений каждого тестового случая.
 */
inline constexpr int NUM_REPEATS = 5;

/**
 * @brief Количество предварительных запусков для прогрева GPU.
 *
 * @details Прогрев выполняется перед серией измерений для уменьшения
 *          влияния первоначальных накладных расходов запуска.
 */
inline constexpr int NUM_WARMUP = 1;

/**
 * @brief Набор размеров квадратных матриц для бенчмаркинга.
 *
 * @details Каждый элемент задаёт размер матриц A, B и C:
 *          A(N x N), B(N x N), C(N x N).
 */
inline constexpr int MATRIX_SIZES[] = {
    256, 512, 1024, 2048, 4096
};

/**
 * @brief Количество размеров матриц в MATRIX_SIZES.
 */
inline constexpr std::size_t NUM_MATRIX_SIZES =
    sizeof(MATRIX_SIZES) / sizeof(MATRIX_SIZES[0]);
