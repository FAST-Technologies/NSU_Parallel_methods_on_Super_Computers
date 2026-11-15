
/**
 * @file matrix_kernels.hpp
 * @brief Объявления CUDA-ядер и функций работы с матрицами.
 */

#pragma once

#include "config.hpp"

#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * @brief Наивное умножение матриц с использованием глобальной памяти GPU.
 *
 * @details Каждый CUDA-поток вычисляет один элемент результирующей матрицы.
 *          Для вычисления элемента выполняется последовательное суммирование
 *          произведений соответствующих элементов строки A и столбца B.
 *
 * @param A Указатель на матрицу A размером rows x N в глобальной памяти GPU.
 * @param B Указатель на матрицу B размером N x N в глобальной памяти GPU.
 * @param C Указатель на результирующую матрицу C размером rows x N.
 * @param N Размер стороны квадратной матрицы B и количество столбцов A.
 * @param rows Количество строк матрицы A, обрабатываемых текущим запуском.
 *
 * @note При потоковом выполнении A и C могут представлять отдельные
 *       фрагменты исходных матриц.
 */
__global__ void matrixMultNaive(
    const BASE_TYPE* A,
    const BASE_TYPE* B,
    BASE_TYPE* C,
    int N,
    int rows
);

/**
 * @brief Блочное умножение матриц с использованием динамической shared memory.
 *
 * @details Матрицы разбиваются на тайлы размером BLOCK_SIZE x BLOCK_SIZE.
 *          Потоки блока кооперативно загружают фрагменты матриц A и B
 *          в разделяемую память, синхронизируются и выполняют вычисления
 *          с повторным использованием загруженных данных.
 *
 *          Размер динамической разделяемой памяти задаётся при запуске ядра.
 *
 * @param A Указатель на матрицу A размером rows x N в глобальной памяти GPU.
 * @param B Указатель на матрицу B размером N x N в глобальной памяти GPU.
 * @param C Указатель на результирующую матрицу C размером rows x N.
 * @param N Размер стороны квадратной матрицы B.
 * @param rows Количество обрабатываемых строк матрицы A.
 *
 * @note Размер выделяемой shared memory должен соответствовать
 *       конфигурации блока и типу BASE_TYPE.
 */
__global__ void matrixMultTiledDynamic(
    const BASE_TYPE* A,
    const BASE_TYPE* B,
    BASE_TYPE* C,
    int N,
    int rows
);

/**
 * @brief Блочное умножение матриц с использованием статической shared memory.
 *
 * @details Каждый CUDA-блок вычисляет тайл результирующей матрицы.
 *          Потоки совместно загружают соответствующие фрагменты матриц A и B
 *          в статически выделенную разделяемую память.
 *
 *          После синхронизации потоков выполняется вычисление частичных сумм.
 *          Для обработки матриц, размеры которых не кратны размеру тайла,
 *          используются проверки границ и заполнение недопустимых элементов
 *          нулевыми значениями.
 *
 * @tparam Tile Размер стороны тайла и соответствующего двумерного блока.
 *
 * @param A Указатель на матрицу A размером rows x N в глобальной памяти GPU.
 * @param B Указатель на матрицу B размером N x N в глобальной памяти GPU.
 * @param C Указатель на результирующую матрицу C размером rows x N.
 * @param N Размер стороны квадратной матрицы B.
 * @param rows Количество обрабатываемых строк матрицы A.
 *
 * @note Размер CUDA-блока при запуске должен соответствовать Tile x Tile.
 */
template <int Tile>
__global__ void matrixMultTiledStaticTpl(
    const BASE_TYPE* A,
    const BASE_TYPE* B,
    BASE_TYPE* C,
    int N,
    int rows
) {
    static_assert(Tile > 0, "Tile must be positive");

    __shared__ BASE_TYPE shared_A[Tile][Tile];
    __shared__ BASE_TYPE shared_B[Tile][Tile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * Tile + tx;
    const int localRow = blockIdx.y * Tile + ty;

    BASE_TYPE sum = BASE_TYPE{0};

    const int numTiles = (N + Tile - 1) / Tile;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int aCol = tile * Tile + tx;
        const int bRow = tile * Tile + ty;

        // Загрузка тайла матрицы A.
        if (localRow < rows && aCol < N) {
            shared_A[ty][tx] = A[localRow * N + aCol];
        } else {
            shared_A[ty][tx] = BASE_TYPE{0};
        }

        // Загрузка тайла матрицы B.
        if (bRow < N && col < N) {
            shared_B[ty][tx] = B[bRow * N + col];
        } else {
            shared_B[ty][tx] = BASE_TYPE{0};
        }

        __syncthreads();

        // Вычисление частичной суммы.
        #pragma unroll
        for (int k = 0; k < Tile; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        // Все потоки должны закончить чтение до перезаписи тайла.
        __syncthreads();
    }

    if (localRow < rows && col < N) {
        C[localRow * N + col] = sum;
    }
}

#endif // __CUDACC__

/**
 * @brief Инициализирует матрицу случайными значениями.
 *
 * @param[out] mat Вектор, представляющий матрицу размером N x N.
 * @param N Размер стороны квадратной матрицы.
 *
 * @note После выполнения функция должна содержать N * N элементов.
 */
void initMatrix(std::vector<BASE_TYPE>& mat, int N);

/**
 * @brief Проверяет корректность результата умножения матриц.
 *
 * @details Сравнивает вычисленную матрицу с эталонной и проверяет,
 *          укладываются ли различия элементов в заданные допуски.
 *
 * @param result Результирующая матрица, полученная GPU-алгоритмом.
 * @param reference Эталонная матрица для сравнения.
 * @param N Размер стороны квадратных матриц.
 *
 * @return true, если проверка всех элементов пройдена;
 *         false, если обнаружено отклонение за пределами допуска.
 */
bool verifyResult(
    const std::vector<BASE_TYPE>& result,
    const std::vector<BASE_TYPE>& reference,
    int N
);

/**
 * @brief Выполняет последовательное умножение матриц на CPU.
 *
 * @details Используется для получения эталонного результата,
 *          с которым сравниваются результаты GPU-вычислений.
 *
 * @param A Исходная матрица A размером N x N.
 * @param B Исходная матрица B размером N x N.
 * @param[out] C Вектор для записи результата умножения размером N x N.
 * @param N Размер стороны квадратных матриц.
 */
void computeCPU(
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    std::vector<BASE_TYPE>& C,
    int N
);
