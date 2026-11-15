
/**
 * @file matrix_kernels.cu
 * @brief Реализация CUDA-ядер и вспомогательных функций
 *        для умножения матриц.
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 15.11.2026.
 * @version 2.0.
 */

#include "matrix_kernels.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

// ============================================================
// 1. Наивный алгоритм (Global Memory)
// ============================================================

/**
 * @brief Выполняет наивное умножение матриц с использованием глобальной памяти GPU.
 *
 * Каждый CUDA-поток вычисляет один элемент выходной матрицы C.
 * Для вычисления скалярного произведения строки A и столбца B
 * поток последовательно читает элементы из глобальной памяти.
 *
 * Матрицы хранятся в row-major формате:
 * - A имеет размер rows x N;
 * - B имеет размер N x N;
 * - C имеет размер rows x N.
 *
 * Параметр rows позволяет обрабатывать как целую матрицу,
 * так и отдельный блок строк при потоковом выполнении.
 *
 * @param A Указатель на входную матрицу A в памяти GPU.
 * @param B Указатель на входную матрицу B в памяти GPU.
 * @param C Указатель на выходную матрицу C в памяти GPU.
 * @param N Размерность квадратной матрицы B и число столбцов A.
 * @param rows Количество строк матриц A и C, обрабатываемых за один запуск.
 */
__global__ void matrixMultNaive(
    const BASE_TYPE* A,
    const BASE_TYPE* B,
    BASE_TYPE* C,
    int N,
    int rows
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int localRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (localRow >= rows || col >= N) {
        return;
    }

    BASE_TYPE sum = BASE_TYPE{0};

    for (int k = 0; k < N; ++k) {
        sum += A[localRow * N + k] * B[k * N + col];
    }

    C[localRow * N + col] = sum;
}

// ============================================================
// 2. Блочный алгоритм (Dynamic Shared Memory)
// ============================================================

/**
 * @brief Выполняет блочное умножение матриц с динамической shared memory.
 *
 * Каждый CUDA-блок вычисляет тайл выходной матрицы.
 * Тайлы входных матриц A и B поочерёдно загружаются
 * из глобальной памяти в динамическую разделяемую память.
 *
 * После синхронизации потоков блока выполняется вычисление
 * частичных скалярных произведений. Граничные элементы,
 * выходящие за пределы матриц, заполняются нулями.
 *
 * Размер динамической shared memory задаётся при запуске ядра
 * и должен быть равен двум тайлам размера TILE_SIZE x TILE_SIZE,
 * умноженным на sizeof(BASE_TYPE).
 *
 * @param A Указатель на входную матрицу A в памяти GPU.
 * @param B Указатель на входную матрицу B в памяти GPU.
 * @param C Указатель на выходную матрицу C в памяти GPU.
 * @param N Размерность квадратной матрицы B и число столбцов A.
 * @param rows Количество обрабатываемых строк матриц A и C.
 */
__global__ void matrixMultTiledDynamic(
    const BASE_TYPE* A,
    const BASE_TYPE* B,
    BASE_TYPE* C,
    int N,
    int rows
) {
    extern __shared__ BASE_TYPE sharedMem[];

    const int tileSize = blockDim.x;

    BASE_TYPE* shared_A = sharedMem;
    BASE_TYPE* shared_B = sharedMem + tileSize * tileSize;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * tileSize + tx;
    const int localRow = blockIdx.y * tileSize + ty;

    BASE_TYPE sum = BASE_TYPE{0};

    const int numTiles = (N + tileSize - 1) / tileSize;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int aCol = tile * tileSize + tx;
        const int bRow = tile * tileSize + ty;

        // Загрузка тайла A.
        if (localRow < rows && aCol < N) {
            shared_A[ty * tileSize + tx] =
                A[localRow * N + aCol];
        } else {
            shared_A[ty * tileSize + tx] = BASE_TYPE{0};
        }

        // Загрузка тайла B.
        if (bRow < N && col < N) {
            shared_B[ty * tileSize + tx] =
                B[bRow * N + col];
        } else {
            shared_B[ty * tileSize + tx] = BASE_TYPE{0};
        }

        __syncthreads();

        // Вычисление частичной суммы.
        for (int k = 0; k < tileSize; ++k) {
            sum += shared_A[ty * tileSize + k] *
                   shared_B[k * tileSize + tx];
        }

        // Синхронизация перед загрузкой следующего тайла.
        __syncthreads();
    }

    if (localRow < rows && col < N) {
        C[localRow * N + col] = sum;
    }
}

// ============================================================
// 3. Инициализация матриц
// ============================================================

/**
 * @brief Заполняет матрицу псевдослучайными значениями.
 *
 * Элементы генерируются в диапазоне [-5, 5].
 * Если размер переданного вектора не соответствует N * N,
 * его размер изменяется до требуемого.
 *
 * Генератор случайных чисел использует стандартную функцию rand().
 * Начальное состояние генератора задаётся вызывающей стороной.
 *
 * @param mat Вектор, в котором хранится матрица.
 * @param N Размерность квадратной матрицы.
 *
 * @throws std::invalid_argument Если N не является положительным числом.
 */
void initMatrix(std::vector<BASE_TYPE>& mat, int N) {
    if (N <= 0) {
        throw std::invalid_argument(
            "Matrix dimension must be positive"
        );
    }

    const std::size_t expectedSize =
        static_cast<std::size_t>(N) * N;

    if (mat.size() != expectedSize) {
        mat.resize(expectedSize);
    }

    for (std::size_t i = 0; i < mat.size(); ++i) {
        const double randomValue =
            static_cast<double>(std::rand()) /
            static_cast<double>(RAND_MAX);

        mat[i] = static_cast<BASE_TYPE>(
            randomValue * 10.0 - 5.0
        );
    }
}

// ============================================================
// 4. Проверка результатов
// ============================================================

/**
 * @brief Проверяет корректность результата умножения матриц.
 *
 * Сравнивает вычисленную матрицу с эталонной.
 * Для сравнения используется комбинированный абсолютный
 * и относительный допуск, значения которых зависят от BASE_TYPE.
 *
 * Дополнительно проверяются размеры матриц и наличие
 * конечных числовых значений. При обнаружении ошибки
 * в консоль выводится информация о первом несовпадающем элементе.
 *
 * @param result Проверяемая матрица.
 * @param reference Эталонная матрица.
 * @param N Размерность квадратной матрицы.
 *
 * @return true, если все элементы удовлетворяют заданному допуску;
 *         false в противном случае.
 */
bool verifyResult(
    const std::vector<BASE_TYPE>& result,
    const std::vector<BASE_TYPE>& reference,
    int N
) {
    if (N <= 0) {
        std::cerr << "Verification failed: invalid matrix dimension\n";
        return false;
    }

    const std::size_t expectedSize =
        static_cast<std::size_t>(N) * N;

    if (result.size() != expectedSize ||
        reference.size() != expectedSize) {

        std::cerr << "Verification failed: matrix size mismatch\n";
        return false;
    }

    // Допуски зависят от выбранной точности.
    constexpr double epsilon_abs =
        std::is_same_v<BASE_TYPE, float> ? 1e-3 : 1e-10;

    constexpr double epsilon_rel =
        std::is_same_v<BASE_TYPE, float> ? 3e-5 : 1e-10;

    for (std::size_t i = 0; i < expectedSize; ++i) {
        const double gpuValue =
            static_cast<double>(result[i]);

        const double referenceValue =
            static_cast<double>(reference[i]);

        if (!std::isfinite(gpuValue) ||
            !std::isfinite(referenceValue)) {

            std::cerr
                << "Verification failed: non-finite value at index "
                << i << '\n';

            return false;
        }

        const double difference =
            std::fabs(gpuValue - referenceValue);

        const double tolerance =
            epsilon_abs +
            epsilon_rel * std::fabs(referenceValue);

        if (difference > tolerance) {
            std::cerr
                << "Verification failed at index " << i
                << ": result = " << gpuValue
                << ", reference = " << referenceValue
                << ", difference = " << difference
                << ", tolerance = " << tolerance
                << '\n';

            return false;
        }
    }

    return true;
}

// ============================================================
// 5. Последовательное умножение матриц на CPU
// ============================================================

/**
 * @brief Вычисляет произведение двух матриц последовательным алгоритмом на CPU.
 *
 * Используется классический алгоритм тройного вложенного цикла.
 * Для накопления суммы произведений применяется тип double,
 * после чего результат преобразуется к BASE_TYPE.
 *
 * Функция предназначена для получения эталонной матрицы
 * при проверке результатов CUDA-вычислений.
 *
 * @param A Первая входная матрица размера N x N.
 * @param B Вторая входная матрица размера N x N.
 * @param C Выходной вектор для матрицы результата.
 * @param N Размерность квадратных матриц.
 *
 * @throws std::invalid_argument Если N некорректен
 *         или размеры входных матриц не соответствуют N * N.
 */
void computeCPU(
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    std::vector<BASE_TYPE>& C,
    int N
) {
    if (N <= 0) {
        throw std::invalid_argument(
            "Matrix dimension must be positive"
        );
    }

    const std::size_t expectedSize =
        static_cast<std::size_t>(N) * N;

    if (A.size() != expectedSize ||
        B.size() != expectedSize) {

        throw std::invalid_argument(
            "Input matrix size mismatch"
        );
    }

    C.resize(expectedSize);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;

            for (int k = 0; k < N; ++k) {
                sum +=
                    static_cast<double>(A[i * N + k]) *
                    static_cast<double>(B[k * N + j]);
            }

            C[i * N + j] = static_cast<BASE_TYPE>(sum);
        }
    }
}
