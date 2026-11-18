
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
#include <algorithm>
#include <cstdint>

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
namespace {

constexpr double kMachineEpsilon =
    std::is_same_v<BASE_TYPE, float>
        ? 1.1920928955078125e-7
        : 2.2204460492503131e-16;

constexpr double kToleranceSafetyFactor = 8.0;

double verificationTolerance(double referenceRms, int N)
{
    return kToleranceSafetyFactor
         * kMachineEpsilon
         * std::sqrt(static_cast<double>(N))
         * std::max(referenceRms, 1.0);
}

/**
 * @brief Проверяет полные строки результата: первую, среднюю, последнюю
 *        и границы чанков потоковых стратегий.
 *
 * @details Эталон строки считается в double по схеме строка A × B с проходом
 *          по строкам B (последовательный доступ), O(N^2) на строку.
 */
bool verifyRows(
    const std::vector<BASE_TYPE>& result,
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    int N)
{
    std::vector<int> rows = {0, N / 2, N - 1};
    const int chunk = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    for (int c = 0; c < NUM_STREAMS; ++c) {
        const int first = c * chunk;
        if (first >= N) break;
        rows.push_back(first);
        rows.push_back(std::min(N, first + chunk) - 1);
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

    const std::size_t n = static_cast<std::size_t>(N);
    std::vector<std::vector<double>> reference;
    reference.reserve(rows.size());
    double sumSq = 0.0;

    for (int row : rows) {
        std::vector<double> acc(n, 0.0);
        for (std::size_t k = 0; k < n; ++k) {
            const double a = static_cast<double>(
                A[static_cast<std::size_t>(row) * n + k]);
            const BASE_TYPE* bRow = B.data() + k * n;
            for (std::size_t col = 0; col < n; ++col) {
                acc[col] += a * static_cast<double>(bRow[col]);
            }
        }
        for (double v : acc) sumSq += v * v;
        reference.push_back(std::move(acc));
    }

    const double rms = std::sqrt(
        sumSq / (static_cast<double>(rows.size()) * static_cast<double>(n)));
    const double tolerance = verificationTolerance(rms, N);

    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::size_t col = 0; col < n; ++col) {
            const double gpu = static_cast<double>(
                result[static_cast<std::size_t>(rows[r]) * n + col]);
            const double diff = std::fabs(gpu - reference[r][col]);
            if (!(diff <= tolerance)) {
                std::cerr << "Row verification failed at (" << rows[r] << ", "
                          << col << "): result = " << gpu
                          << ", reference = " << reference[r][col]
                          << ", difference = " << diff
                          << ", tolerance = " << tolerance << '\n';
                return false;
            }
        }
    }
    return true;
}

} // namespace


/**
 * @brief Проверяет корректность результата умножения матриц.
 *
 * Сравнивает вычисленную матрицу с эталонной.
 * Для сравнения используется единый абсолютный допуск, масштабируемый по RMS эталона и √N, 
 * значения которых зависят от BASE_TYPE.
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

    double referenceSumSq = 0.0;

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

        referenceSumSq += referenceValue * referenceValue;
    }

    const double referenceRms =
        std::sqrt(referenceSumSq / static_cast<double>(expectedSize));

    const double tolerance =
        verificationTolerance(referenceRms, N);

    for (std::size_t i = 0; i < expectedSize; ++i) {
        const double gpuValue =
            static_cast<double>(result[i]);

        const double referenceValue =
            static_cast<double>(reference[i]);

        const double difference =
            std::fabs(gpuValue - referenceValue);

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

bool verifySampled(
    const std::vector<BASE_TYPE>& result,
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    int N,
    int samples
) {
    if (N <= 0) {
        std::cerr << "Verification failed: invalid matrix dimension\n";
        return false;
    }

    const std::size_t expectedSize =
        static_cast<std::size_t>(N) * N;

    if (result.size() != expectedSize ||
        A.size() != expectedSize ||
        B.size() != expectedSize) {
        std::cerr << "Verification failed: matrix size mismatch\n";
        return false;
    }

    for (std::size_t i = 0; i < expectedSize; ++i) {
        if (!std::isfinite(static_cast<double>(result[i]))) {
            std::cerr
                << "Verification failed: non-finite result at index "
                << i << '\n';
            return false;
        }
    }

    struct Point {
        int row;
        int col;
    };

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(std::max(samples, 0)) + 4);

    points.push_back({0, 0});
    points.push_back({0, N - 1});
    points.push_back({N - 1, 0});
    points.push_back({N - 1, N - 1});

    // Детерминированная псевдослучайная выборка.
    std::uint64_t state = 123456789ULL;

    const auto nextIndex = [&state, N]() -> int {
        state = state * 6364136223846793005ULL
                      + 1442695040888963407ULL;

        return static_cast<int>(
            state % static_cast<std::uint64_t>(N)
        );
    };

    for (int s = 0; s < std::max(samples, 0); ++s) {
        const int row = nextIndex();
        const int col = nextIndex();

        points.push_back({row, col});
    }

    std::vector<double> referenceValues(points.size());
    double referenceSumSq = 0.0;

    // Независимый CPU-эталон
    for (std::size_t p = 0; p < points.size(); ++p) {
        const int row = points[p].row;
        const int col = points[p].col;

        double sum = 0.0;

        for (int k = 0; k < N; ++k) {
            const double a =
                static_cast<double>(
                    A[static_cast<std::size_t>(row) * N + k]
                );

            const double b =
                static_cast<double>(
                    B[static_cast<std::size_t>(k) * N + col]
                );

            sum += a * b;
        }

        if (!std::isfinite(sum)) {
            std::cerr
                << "Verification failed: non-finite CPU reference at ("
                << row << ", " << col << ")\n";
            return false;
        }

        referenceValues[p] = sum;
        referenceSumSq += sum * sum;
    }

    const double referenceRms =
        std::sqrt(
            referenceSumSq / static_cast<double>(points.size())
        );

    const double tolerance =
        verificationTolerance(referenceRms, N);

    for (std::size_t p = 0; p < points.size(); ++p) {
        const int row = points[p].row;
        const int col = points[p].col;

        const std::size_t index =
            static_cast<std::size_t>(row) * N + col;

        const double gpuValue =
            static_cast<double>(result[index]);

        const double referenceValue = referenceValues[p];

        const double difference =
            std::fabs(gpuValue - referenceValue);

        if (difference > tolerance) {
            std::cerr
                << "Verification failed at (" << row << ", " << col << ")"
                << ": result = " << gpuValue
                << ", reference = " << referenceValue
                << ", difference = " << difference
                << ", tolerance = " << tolerance
                << '\n';

            return false;
        }
    }

    return verifyRows(result, A, B, N);
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
