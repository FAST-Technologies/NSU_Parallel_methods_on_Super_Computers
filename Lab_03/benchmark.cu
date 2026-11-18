/**
 * @file benchmark.cu
 * @brief Реализация функций тестирования производительности CUDA-алгоритмов.
 * @details Содержит реализацию запуска алгоритмов умножения матриц,
 * измерения времени выполнения, управления CUDA streams,
 * работы с различными стратегиями памяти и сохранения
 * результатов экспериментов.
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 15.11.2026.
 * @version 2.0.
 */

#include "benchmark.hpp"

#include "config.hpp"
#include "cuda_utils.hpp"
#include "matrix_kernels.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <iomanip>
#include <type_traits>
#include <vector>

// ============================================================
// cuBLAS error handling
// ============================================================
namespace {

/**
 * @brief Возвращает текстовое представление статуса cuBLAS.
 *
 * Используется при выводе диагностических сообщений
 * в случае ошибки вызова библиотеки cuBLAS.
 *
 * @param status Код состояния cuBLAS.
 * @return Строковое имя соответствующего статуса.
 */    
const char* cublasStatusName(cublasStatus_t status)
{
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        const cublasStatus_t status_ = (call);                               \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                              \
            std::cerr << "cuBLAS Error: " << cublasStatusName(status_)       \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

/**
 * @brief Выполняет умножение матриц средствами cuBLAS.
 *
 * Выбирает cublasSgemm для типа float и cublasDgemm
 * для типа double на этапе компиляции.
 *
 * Поскольку исходные матрицы хранятся в row-major формате,
 * операнды передаются в обратном порядке, что позволяет
 * получить требуемый результат с учётом column-major
 * соглашения cuBLAS.
 *
 * @tparam T Тип элементов матриц: float или double.
 * @param handle Инициализированный дескриптор cuBLAS.
 * @param N Число столбцов A и размерность B.
 * @param rows Количество обрабатываемых строк A и C.
 * @param d_A Указатель на A в памяти GPU.
 * @param d_B Указатель на B в памяти GPU.
 * @param d_C Указатель на C в памяти GPU.
 */
template <typename T>
void launchCublasGemm(
    cublasHandle_t handle,
    int N,
    int rows,
    const T* d_A,
    const T* d_B,
    T* d_C)
{
    const T alpha = T{1};
    const T beta  = T{0};

    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, rows, N,
            &alpha,
            d_B, N,
            d_A, N,
            &beta,
            d_C, N
        ));
    }
    else if constexpr (std::is_same_v<T, double>) {
        CUBLAS_CHECK(cublasDgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, rows, N,
            &alpha,
            d_B, N,
            d_A, N,
            &beta,
            d_C, N
        ));
    }
    else {
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, double>,
            "cuBLAS supports only float and double in this benchmark"
        );
    }
}

using Clock = std::chrono::steady_clock;

/**
 * @brief Вычисляет продолжительность интервала времени в миллисекундах.
 *
 * Использует монотонные часы std::chrono::steady_clock.
 *
 * @param start Начальная временная отметка.
 * @param end Конечная временная отметка.
 * @return Продолжительность интервала в миллисекундах.
 */
double elapsedMilliseconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Получает длительность CUDA-операции по двум событиям.
 *
 * Предполагается, что оба события уже записаны в CUDA stream,
 * а операция между ними завершена.
 *
 * @param start Начальное CUDA-событие.
 * @param stop Конечное CUDA-событие.
 * @return Время между событиями в миллисекундах.
 */
double eventElapsed(cudaEvent_t start, cudaEvent_t stop)
{
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds);
}

/**
 * @brief Разделяемый cuBLAS-handle: создаётся один раз, чтобы cublasCreate
 *        и ленивая инициализация не попадали в каждое измерение.
 */
cublasHandle_t sharedHandle()
{
    struct Holder {
        cublasHandle_t h = nullptr;
        Holder() { CUBLAS_CHECK(cublasCreate(&h)); }
        ~Holder() { if (h != nullptr) { cublasDestroy(h); } }
    };
    static Holder holder;
    return holder.h;
}

/** @brief Пул handle'ов, по одному на CUDA stream (создаётся один раз). */
std::vector<cublasHandle_t>& streamHandlePool(int count)
{
    struct Pool {
        std::vector<cublasHandle_t> h;
        ~Pool() { for (auto x : h) { if (x != nullptr) { cublasDestroy(x); } } }
    };
    static Pool pool;
    while (static_cast<int>(pool.h.size()) < count) {
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        pool.h.push_back(handle);
    }
    return pool.h;
}

/**
 * @brief Prefetch managed-памяти на GPU (toCpu=false) или на CPU (toCpu=true).
 *
 * @details Сигнатура cudaMemPrefetchAsync зависит от версии CUDA, а не от ОС:
 *          в CUDA 13 принимается cudaMemLocation + flags.
 */
cudaError_t prefetchManaged(const void* ptr, std::size_t bytes, bool toCpu)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation location{};
    location.type = toCpu ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    location.id   = toCpu ? 0 : device;
    return cudaMemPrefetchAsync(ptr, bytes, location, 0, nullptr);
#else
    return cudaMemPrefetchAsync(ptr, bytes, toCpu ? cudaCpuDeviceId : device, nullptr);
#endif
}

// Текущий размер блока; по умолчанию берётся из config.hpp. 
int g_blockSize = TILE_SIZE; 

template <int Tile> void launchStaticTiled( 
    dim3 grid, 
    cudaStream_t stream, 
    const BASE_TYPE* d_A, 
    const BASE_TYPE* d_B, 
    BASE_TYPE* d_C, 
    int N, 
    int rows
) 
{ 
    matrixMultTiledStaticTpl<Tile><<<grid, dim3(Tile, Tile), 0, stream>>>( d_A, d_B, d_C, N, rows ); 
}

/**
 * @brief Выполняет умножение матриц выбранным алгоритмом.
 *
 * @param algorithm Выбранный алгоритм.
 * @param d_A Указатель на матрицу A в памяти GPU.
 * @param d_B Указатель на матрицу B в памяти GPU.
 * @param d_C Указатель на матрицу результата в памяти GPU.
 * @param N Размер квадратной матрицы B и число столбцов A.
 * @param rows Число обрабатываемых строк матриц A и C.
 * @param stream CUDA stream, в котором выполняется операция.
 * @param handle cuBLAS handle; используется только для алгоритма cuBLAS.
 */
void launchAlgorithm(
    Algorithm algorithm,
    const BASE_TYPE* d_A,
    const BASE_TYPE* d_B,
    BASE_TYPE* d_C,
    int N,
    int rows,
    cudaStream_t stream,
    cublasHandle_t handle = nullptr)
{
    const int bs = g_blockSize;
    const dim3 block(bs, bs);
    const dim3 grid(
        (N + bs - 1) / bs,
        (rows + bs - 1) / bs
    );

    switch (algorithm) {
        case Algorithm::Naive:
            matrixMultNaive<<<grid, block, 0, stream>>>(
                d_A, d_B, d_C, N, rows
            );
            break;

        case Algorithm::StaticTiled: 
            switch (bs) { 
                case 2: launchStaticTiled<2>(grid, stream, d_A, d_B, d_C, N, rows); break; 
                case 4: launchStaticTiled<4>(grid, stream, d_A, d_B, d_C, N, rows); break; 
                case 8: launchStaticTiled<8>(grid, stream, d_A, d_B, d_C, N, rows); break; 
                case 16: launchStaticTiled<16>(grid, stream, d_A, d_B, d_C, N, rows); break; 
                case 32: launchStaticTiled<32>(grid, stream, d_A, d_B, d_C, N, rows); break; 
                default: 
                    std::cerr << "Unsupported block size: " << bs << '\n'; 
                    std::exit(EXIT_FAILURE); 
                } 
                break;

        case Algorithm::DynamicTiled:
            matrixMultTiledDynamic<<<
                grid,
                block,
                2 * bs * bs * sizeof(BASE_TYPE),
                stream
            >>>(d_A, d_B, d_C, N, rows);
            break;

        case Algorithm::cuBLAS:
            if (handle == nullptr) {
                std::cerr << "cuBLAS handle is null\n";
                std::exit(EXIT_FAILURE);
            }

            launchCublasGemm<BASE_TYPE>(
                handle, N, rows, d_A, d_B, d_C
            );
            break;
    }

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Выполняет умножение матриц, полностью размещая рабочие данные
 *        в соответствующих буферах памяти GPU.
 *
 * Поддерживаются стратегии Pageable, Pinned, UnifiedPrefetch
 * и UnifiedManaged.
 *
 * Измеряются время передачи входных данных на GPU (H2D),
 * время выполнения алгоритма и время получения результата (D2H).
 * Полное время включает выделение pinned-буферов и копирование данных в них и 
 * обратно (для Pinned), выделение и освобождение ресурсов GPU. Создание cuBLAS-handle не входит.
 *
 * @param algorithm Алгоритм умножения матриц.
 * @param strategy Стратегия управления памятью.
 * @param A Входная матрица A размера N x N.
 * @param B Входная матрица B размера N x N.
 * @param C Выходная матрица C размера N x N.
 * @param N Размерность квадратных матриц.
 * @return Структура с измеренными временами выполнения стадий.
 */
Timing runFullMatrix(
    Algorithm algorithm,
    MemoryStrategy strategy,
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    std::vector<BASE_TYPE>& C,
    int N)
{
    Timing timing{};

    cublasHandle_t handle = nullptr;
    if (algorithm == Algorithm::cuBLAS) {
        handle = sharedHandle();
    }

    const auto totalStart = Clock::now();

    const std::size_t matrixElements =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    const std::size_t matrixBytes = matrixElements * sizeof(BASE_TYPE);

    const bool usePinned = (strategy == MemoryStrategy::Pinned);
    const bool useManaged =
        strategy == MemoryStrategy::UnifiedPrefetch ||
        strategy == MemoryStrategy::UnifiedManaged;

    BASE_TYPE* d_A = nullptr;
    BASE_TYPE* d_B = nullptr;
    BASE_TYPE* d_C = nullptr;

    BASE_TYPE* h_A = const_cast<BASE_TYPE*>(A.data());
    BASE_TYPE* h_B = const_cast<BASE_TYPE*>(B.data());
    BASE_TYPE* h_C = C.data();

    BASE_TYPE* pinned_A = nullptr;
    BASE_TYPE* pinned_B = nullptr;
    BASE_TYPE* pinned_C = nullptr;

    if (usePinned) {
        CUDA_CHECK(cudaMallocHost(&pinned_A, matrixBytes));
        CUDA_CHECK(cudaMallocHost(&pinned_B, matrixBytes));
        CUDA_CHECK(cudaMallocHost(&pinned_C, matrixBytes));

        std::copy(A.begin(), A.end(), pinned_A);
        std::copy(B.begin(), B.end(), pinned_B);

        h_A = pinned_A;
        h_B = pinned_B;
        h_C = pinned_C;
    }

    if (useManaged) {
        CUDA_CHECK(cudaMallocManaged(&d_A, matrixBytes));
        CUDA_CHECK(cudaMallocManaged(&d_B, matrixBytes));
        CUDA_CHECK(cudaMallocManaged(&d_C, matrixBytes));

        const auto fillStart = Clock::now();
        std::copy(A.begin(), A.end(), d_A);
        std::copy(B.begin(), B.end(), d_B);
        timing.h2d = elapsedMilliseconds(fillStart, Clock::now());

        if (strategy == MemoryStrategy::UnifiedPrefetch) {
            const auto prefetchStart = Clock::now();

            CUDA_CHECK(prefetchManaged(d_A, matrixBytes, false));
            CUDA_CHECK(prefetchManaged(d_B, matrixBytes, false));
            CUDA_CHECK(prefetchManaged(d_C, matrixBytes, false));

            CUDA_CHECK(cudaDeviceSynchronize());

            timing.h2d += elapsedMilliseconds(prefetchStart, Clock::now());
        }
    }
    else {
        CUDA_CHECK(cudaMalloc(&d_A, matrixBytes));
        CUDA_CHECK(cudaMalloc(&d_B, matrixBytes));
        CUDA_CHECK(cudaMalloc(&d_C, matrixBytes));

        const auto transferStart = Clock::now();

        CUDA_CHECK(cudaMemcpy(
            d_A, h_A, matrixBytes, cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_B, h_B, matrixBytes, cudaMemcpyHostToDevice
        ));

        timing.h2d = elapsedMilliseconds(
            transferStart, Clock::now()
        );
    }

    cudaEvent_t kernelStart{}, kernelStop{};
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));

    CUDA_CHECK(cudaEventRecord(kernelStart));

    launchAlgorithm(
        algorithm,
        d_A, d_B, d_C,
        N, N,
        nullptr,
        handle
    );

    CUDA_CHECK(cudaEventRecord(kernelStop));
    CUDA_CHECK(cudaEventSynchronize(kernelStop));

    timing.kernel = eventElapsed(kernelStart, kernelStop);

    const auto downloadStart = Clock::now();

    if (strategy == MemoryStrategy::UnifiedPrefetch) {
        CUDA_CHECK(prefetchManaged(d_C, matrixBytes, true));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::copy(d_C, d_C + matrixElements, C.begin());
        timing.d2h = elapsedMilliseconds(
            downloadStart, Clock::now()
        );
    }
    else if (strategy == MemoryStrategy::UnifiedManaged) {
        CUDA_CHECK(cudaDeviceSynchronize());
        std::copy(d_C, d_C + matrixElements, C.begin());
        timing.d2h = elapsedMilliseconds(
            downloadStart, Clock::now()
        );
    }
    else {
        CUDA_CHECK(cudaMemcpy(
            h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost
        ));

        if (usePinned) {
            std::copy(pinned_C, pinned_C + matrixElements, C.begin());
        }

        timing.d2h = elapsedMilliseconds(
            downloadStart, Clock::now()
        );
    }

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    if (usePinned) {
        CUDA_CHECK(cudaFreeHost(pinned_A));
        CUDA_CHECK(cudaFreeHost(pinned_B));
        CUDA_CHECK(cudaFreeHost(pinned_C));
    }

    timing.total = elapsedMilliseconds(totalStart, Clock::now());
    return timing;
}

/**
 * @brief Выполняет умножение матриц с разделением A и C на части
 *        и распределением вычислений между несколькими CUDA streams.
 *
 * Матрица B передаётся на GPU один раз и используется всеми streams.
 * Матрицы A и C обрабатываются блоками строк.
 * Для каждой части выполняются передача входных данных,
 * вычисление и обратная передача результата.
 *
 * Времена H2D, Kernel и D2H представляют максимальные длительности
 * соответствующих стадий среди активных streams. Эти стадии могут
 * выполняться параллельно, поэтому их значения нельзя складывать
 * для получения полного времени.
 *
 * Полное время измеряется отдельно и включает создание streams,
 * выделение буферов, передачу B, обработку частей и освобождение
 * ресурсов. 
 *
 * @param algorithm Алгоритм умножения матриц.
 * @param strategy Стратегия потоковой передачи данных.
 * @param A Входная матрица A размера N x N.
 * @param B Входная матрица B размера N x N.
 * @param C Выходная матрица C размера N x N.
 * @param N Размерность квадратных матриц.
 * @return Структура с измеренными временами выполнения стадий.
 */
Timing runStreaming(
    Algorithm algorithm,
    MemoryStrategy strategy,
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    std::vector<BASE_TYPE>& C,
    int N)
{
    Timing timing{};
    const auto totalStart = Clock::now();

    const bool usePinned =
        strategy == MemoryStrategy::PinnedStreams;

    const std::size_t matrixBytes =
        static_cast<std::size_t>(N) * N * sizeof(BASE_TYPE);

    const int streamCount = std::min(NUM_STREAMS, N);
    const int rowsPerStream = (N + streamCount - 1) / streamCount;

    BASE_TYPE* d_B = nullptr;
    BASE_TYPE* pinned_B = nullptr;

    const BASE_TYPE* hostB = B.data();

    std::vector<cudaStream_t> streams(streamCount);
    std::vector<cublasHandle_t> handles(streamCount, nullptr);
    if (algorithm == Algorithm::cuBLAS) {
        auto& pool = streamHandlePool(streamCount);
        for (int i = 0; i < streamCount; ++i) {
            handles[i] = pool[i];
        }
    }

    std::vector<BASE_TYPE*> d_A(streamCount, nullptr);
    std::vector<BASE_TYPE*> d_C(streamCount, nullptr);

    std::vector<BASE_TYPE*> pinned_A(streamCount, nullptr);
    std::vector<BASE_TYPE*> pinned_C(streamCount, nullptr);

    std::vector<cudaEvent_t> h2dStart(streamCount);
    std::vector<cudaEvent_t> h2dStop(streamCount);
    std::vector<cudaEvent_t> kernelStart(streamCount);
    std::vector<cudaEvent_t> kernelStop(streamCount);
    std::vector<cudaEvent_t> d2hStart(streamCount);
    std::vector<cudaEvent_t> d2hStop(streamCount);

    std::vector<int> rowCounts(streamCount);
    std::vector<int> rowOffsets(streamCount);

    if (usePinned) {
        CUDA_CHECK(cudaMallocHost(&pinned_B, matrixBytes));
        std::copy(B.begin(), B.end(), pinned_B);
        hostB = pinned_B;
    }

    // Создание streams, буферов и событий.
    for (int i = 0; i < streamCount; ++i) {
        const int rowOffset = i * rowsPerStream;
        const int rows = std::max(0, std::min(rowsPerStream, N - rowOffset));

        rowOffsets[i] = rowOffset;
        rowCounts[i] = rows;

        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

        CUDA_CHECK(cudaEventCreate(&h2dStart[i]));
        CUDA_CHECK(cudaEventCreate(&h2dStop[i]));
        CUDA_CHECK(cudaEventCreate(&kernelStart[i]));
        CUDA_CHECK(cudaEventCreate(&kernelStop[i]));
        CUDA_CHECK(cudaEventCreate(&d2hStart[i]));
        CUDA_CHECK(cudaEventCreate(&d2hStop[i]));

        if (rows == 0) {
            continue;
        }

        const std::size_t elems = static_cast<std::size_t>(rows) * N;
        if (usePinned) {
            CUDA_CHECK(cudaMallocHost(&pinned_A[i], elems * sizeof(BASE_TYPE)));
            CUDA_CHECK(cudaMallocHost(&pinned_C[i], elems * sizeof(BASE_TYPE)));
            const BASE_TYPE* src = A.data() + static_cast<std::size_t>(rowOffset) * N;
            std::copy(src, src + elems, pinned_A[i]);
        }
        if (algorithm == Algorithm::cuBLAS) {
            CUBLAS_CHECK(cublasSetStream(handles[i], streams[i]));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMalloc(&d_B, matrixBytes));
    for (int i = 0; i < streamCount; ++i) {
        if (rowCounts[i] == 0) continue;
        const std::size_t chunkBytes =
            static_cast<std::size_t>(rowCounts[i]) * N * sizeof(BASE_TYPE);
        CUDA_CHECK(cudaMalloc(&d_A[i], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_C[i], chunkBytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto bUploadStart = Clock::now();

    CUDA_CHECK(cudaMemcpyAsync(
        d_B,
        hostB,
        matrixBytes,
        cudaMemcpyHostToDevice,
        streams[0]
    ));
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    const double bUploadTime = elapsedMilliseconds(
        bUploadStart, Clock::now()
    );

    // Запуск независимых частей вычисления.
    for (int i = 0; i < streamCount; ++i) {
        const int rows = rowCounts[i];
        if (rows == 0) {
            continue;
        }

        const int rowOffset = rowOffsets[i];

        const BASE_TYPE* hostA = usePinned
            ? pinned_A[i]
            : A.data() + static_cast<std::size_t>(rowOffset) * N;

        BASE_TYPE* hostC = usePinned
            ? pinned_C[i]
            : C.data() + static_cast<std::size_t>(rowOffset) * N;

        const std::size_t chunkBytes =
            static_cast<std::size_t>(rows) * N * sizeof(BASE_TYPE);

        CUDA_CHECK(cudaEventRecord(h2dStart[i], streams[i]));

        CUDA_CHECK(cudaMemcpyAsync(
            d_A[i],
            hostA,
            chunkBytes,
            cudaMemcpyHostToDevice,
            streams[i]
        ));

        CUDA_CHECK(cudaEventRecord(h2dStop[i], streams[i]));
        CUDA_CHECK(cudaEventRecord(kernelStart[i], streams[i]));

        launchAlgorithm(
            algorithm,
            d_A[i],
            d_B,
            d_C[i],
            N,
            rows,
            streams[i],
            handles[i]
        );

        CUDA_CHECK(cudaEventRecord(kernelStop[i], streams[i]));
        CUDA_CHECK(cudaEventRecord(d2hStart[i], streams[i]));

        CUDA_CHECK(cudaMemcpyAsync(
            hostC,
            d_C[i],
            chunkBytes,
            cudaMemcpyDeviceToHost,
            streams[i]
        ));

        CUDA_CHECK(cudaEventRecord(d2hStop[i], streams[i]));
    }

    // Ожидание завершения всех streams.
    for (int i = 0; i < streamCount; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    double maxH2D = 0.0;
    double maxKernel = 0.0;
    double maxD2H = 0.0;

    for (int i = 0; i < streamCount; ++i) {
        if (rowCounts[i] == 0) {
            continue;
        }

        maxH2D = std::max(
            maxH2D, eventElapsed(h2dStart[i], h2dStop[i])
        );
        maxKernel = std::max(
            maxKernel, eventElapsed(kernelStart[i], kernelStop[i])
        );
        maxD2H = std::max(
            maxD2H, eventElapsed(d2hStart[i], d2hStop[i])
        );
    }

    timing.h2d = bUploadTime + maxH2D;
    timing.kernel = maxKernel;
    timing.d2h = maxD2H;

    if (usePinned) {
        for (int i = 0; i < streamCount; ++i) {
            if (rowCounts[i] == 0) {
                continue;
            }

            const std::size_t elements =
                static_cast<std::size_t>(rowCounts[i]) * N;

            const std::size_t offset =
                static_cast<std::size_t>(rowOffsets[i]) * N;

            std::copy(
                pinned_C[i],
                pinned_C[i] + elements,
                C.begin() + static_cast<std::ptrdiff_t>(offset)
            );
        }
    }

    // Освобождение ресурсов.
    for (int i = 0; i < streamCount; ++i) {
        if (handles[i] != nullptr) {
            CUBLAS_CHECK(cublasSetStream(handles[i], nullptr));
        }

        if (d_A[i] != nullptr) {
            CUDA_CHECK(cudaFree(d_A[i]));
        }
        if (d_C[i] != nullptr) {
            CUDA_CHECK(cudaFree(d_C[i]));
        }

        if (pinned_A[i] != nullptr) {
            CUDA_CHECK(cudaFreeHost(pinned_A[i]));
        }
        if (pinned_C[i] != nullptr) {
            CUDA_CHECK(cudaFreeHost(pinned_C[i]));
        }

        CUDA_CHECK(cudaEventDestroy(h2dStart[i]));
        CUDA_CHECK(cudaEventDestroy(h2dStop[i]));
        CUDA_CHECK(cudaEventDestroy(kernelStart[i]));
        CUDA_CHECK(cudaEventDestroy(kernelStop[i]));
        CUDA_CHECK(cudaEventDestroy(d2hStart[i]));
        CUDA_CHECK(cudaEventDestroy(d2hStop[i]));

        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    if (pinned_B != nullptr) {
        CUDA_CHECK(cudaFreeHost(pinned_B));
    }

    CUDA_CHECK(cudaFree(d_B));

    timing.total = elapsedMilliseconds(totalStart, Clock::now());
    return timing;
}

} // namespace

// ============================================================
// Публичные функции benchmark.hpp
// ============================================================

/**
 * @brief Возвращает название алгоритма для вывода и сохранения результатов.
 * @param algorithm Идентификатор алгоритма.
 * @return Строковое название алгоритма.
 */
const char* algorithmName(Algorithm algorithm)
{
    switch (algorithm) {
        case Algorithm::Naive:         return "Naive";
        case Algorithm::StaticTiled:   return "Static Tiled";
        case Algorithm::DynamicTiled:  return "Dynamic Tiled";
        case Algorithm::cuBLAS:        return "cuBLAS";
        default:                       return "Unknown";
    }
}

/**
 * @brief Возвращает название стратегии памяти для вывода и CSV.
 * @param strategy Идентификатор стратегии памяти.
 * @return Строковое название стратегии.
 */
const char* strategyName(MemoryStrategy strategy)
{
    switch (strategy) {
        case MemoryStrategy::Pageable:         return "Pageable";
        case MemoryStrategy::Pinned:           return "Pinned";
        case MemoryStrategy::PageableStreams:  return "Pageable Streams";
        case MemoryStrategy::PinnedStreams:    return "Pinned Streams";
        case MemoryStrategy::UnifiedPrefetch:  return "Unified Prefetch";
        case MemoryStrategy::UnifiedManaged:   return "Unified Managed";
        default:                               return "Unknown";
    }
}

/**
 * @brief Проверяет поддержку concurrent managed access текущим GPU.
 *
 * Свойство используется для определения возможности включения
 * стратегии UnifiedPrefetch в набор экспериментов.
 *
 * @return true, если concurrentManagedAccess ненулевой;
 *         false, если свойство отсутствует или сведения об устройстве
 *         получить не удалось.
 */
bool isUnifiedPrefetchSupported()
{
    int device = 0;
    cudaDeviceProp properties{};

    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    return properties.concurrentManagedAccess != 0;
}

/**
 * @brief Выводит информацию о текущем CUDA-устройстве.
 */
void printGPUInfo()
{
    int device = 0;
    cudaDeviceProp properties{};

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    bool result = properties.kernelExecTimeoutEnabled ? "true" : "false";

    std::cout << "\n========== GPU INFORMATION ==========\n";
    std::cout << "Device: " << properties.name << '\n';
    std::cout << "Compute Capability: "
              << properties.major << "." << properties.minor << '\n';
    std::cout << "Global Memory: "
              << properties.totalGlobalMem / (1024.0 * 1024.0)
              << " MB\n";
    std::cout << "Shared memory per block: "
              << properties.sharedMemPerBlock << '\n';
    std::cout << "Registers per block: "
              << properties.regsPerBlock << '\n';
    std::cout << "Warp size: "
              << properties.warpSize << '\n';
    std::cout << "Memory pitch: "
              << properties.memPitch << '\n';
    std::cout << "Multiprocessors: "
              << properties.multiProcessorCount << '\n';
    std::cout << "Max Threads per Block: "
              << properties.maxThreadsPerBlock << '\n';
    std::cout << "Concurrent Managed Access: "
              << properties.concurrentManagedAccess << '\n';
    std::cout << "Pageable Memory Access: "
              << properties.pageableMemoryAccess << '\n';
    std::cout << "Max threads dimensions: "
              << properties.maxThreadsDim[0] << ' ' << properties.maxThreadsDim[1] << ' ' << properties.maxThreadsDim[2] <<'\n';
    std::cout << "Max grid size: "
              << properties.maxGridSize[0] << ' ' << properties.maxGridSize[1] << ' ' << properties.maxGridSize[2] <<'\n';
    std::cout << "Clock rate: "
              << properties.clockRate << '\n'; 
    std::cout << "Total constant memory: "
              << properties.totalConstMem << '\n';          
    std::cout << "Texture alignment: "
              << properties.textureAlignment << '\n';
    std::cout << "Device overlap: "
              << properties.deviceOverlap << '\n';
    std::cout << "Multiprocessor count: "
              << properties.multiProcessorCount << '\n';
    std::cout << "Kernel execution timeout enabled: "
              << result << '\n';
    std::cout << "=====================================\n\n";
}

/**
 * @brief Запускает один эксперимент для выбранного алгоритма и стратегии памяти.
 *
 * @param algorithm Алгоритм умножения матриц.
 * @param strategy Стратегия размещения и передачи данных.
 * @param A Входная матрица A размера N x N.
 * @param B Входная матрица B размера N x N.
 * @param C Выходная матрица C размера N x N.
 * @param N Размер матриц.
 * @return Времена передачи данных, вычисления и полного выполнения.
 */
Timing runOneCase(
    Algorithm algorithm,
    MemoryStrategy strategy,
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    std::vector<BASE_TYPE>& C,
    int N)
{
    if (N <= 0) {
        std::cerr << "Matrix size must be positive\n";
        std::exit(EXIT_FAILURE);
    }

    const std::size_t expectedSize =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(N);

    if (A.size() != expectedSize ||
        B.size() != expectedSize ||
        C.size() != expectedSize) {
        std::cerr << "Invalid matrix buffer size\n";
        std::exit(EXIT_FAILURE);
    }

    switch (strategy) {
        case MemoryStrategy::Pageable:
        case MemoryStrategy::Pinned:
        case MemoryStrategy::UnifiedPrefetch:
        case MemoryStrategy::UnifiedManaged:
            return runFullMatrix(
                algorithm, strategy, A, B, C, N
            );

        case MemoryStrategy::PageableStreams:
        case MemoryStrategy::PinnedStreams:
            return runStreaming(
                algorithm, strategy, A, B, C, N
            );

        default:
            std::cerr << "Unknown memory strategy\n";
            std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Вычисляет медиану набора измерений.
 *
 * Для чётного количества значений возвращает среднее двух центральных
 * элементов отсортированного набора. Для пустого набора возвращает 0.
 *
 * @param values Набор измеренных значений.
 * @return Медиана в миллисекундах либо 0.0 для пустого набора.
 */
double median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }

    return values[middle];
}

/**
 * @brief Возвращает минимальное значение набора измерений.
 *
 * Для пустого набора возвращает 0.0.
 *
 * @param values Набор измеренных значений.
 * @return Минимальное значение либо 0.0 для пустого набора.
 */
double minimum(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }

    return *std::min_element(values.begin(), values.end());
}

/**
 * @brief Записывает строку заголовков CSV-файла.
 * @param file Открытый поток выходного CSV-файла.
 */
void writeCSVHeader(std::ofstream& file)
{
    file << "N,Block,Precision,Strategy,Algorithm,"
         << "H2D_ms,Kernel_ms,D2H_ms,"
         << "TotalMedian_ms,TotalMin_ms,"
         << "GFLOPS_Kernel,GFLOPS_Total,Verified\n";
}

/**
 * @brief Записывает результаты одного эксперимента в CSV-файл.
 * @param file Открытый поток выходного CSV-файла.
 * @param result Структура с параметрами и результатами эксперимента.
 */
void writeCSVRow(
    std::ofstream& file,
    const BenchmarkResult& result)
{
    file << std::fixed << std::setprecision(6)
         << result.N << ','
         << result.blockSize << ','
         << result.precision << ','
         << result.strategy << ','
         << result.algorithm << ','
         << result.h2dMedian << ','
         << result.kernelMedian << ','
         << result.d2hMedian << ','
         << result.totalMedian << ','
         << result.totalMin << ','
         << result.gflopsKernel << ','
         << result.gflopsTotal << ','
         << (result.verified ? "true" : "false")
         << '\n';
}

bool isSupportedBlockSize(int blockSize)
{
    return blockSize == 2 || blockSize == 4 || blockSize == 8 ||
    blockSize == 16 || blockSize == 32;
}

void setBlockSize(int blockSize)
{
    if (!isSupportedBlockSize(blockSize)) {
        std::cerr << "Unsupported block size: " << blockSize << '\n';
        std::exit(EXIT_FAILURE);
    }
    g_blockSize = blockSize;
}

int getBlockSize()
{
    return g_blockSize;
}