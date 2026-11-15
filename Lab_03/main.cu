/**
 * @file main.cu
 * @brief Лабораторная работа №3: cuBLAS Parallel Programming.
 * @details Точка входа в программу. Поддерживает два режима:
 *          1. Бенчмаркинг (без аргументов): тестирование наборов размеров.
 *          2. Одиночный запуск (с аргументами): тестирование конкретных N и размера блока.
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 04.11.2026.
 * @version 2.0 (Dual-mode execution, Hybrid Optimized)
 */

#include "matrix_kernels.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef _WIN32
#include <windows.h>
#endif


// ============================================================
// Configuration
// ============================================================

constexpr int TILE_SIZE = 16;
constexpr int NUM_STREAMS = 4;
constexpr int NUM_REPEATS = 5;
constexpr int NUM_WARMUP = 1;

const std::vector<int> MATRIX_SIZES = {
        256, 512, 1024, 2048, 4096
};


// ============================================================
// Error checking
// ============================================================

#define CUBLAS_CHECK(call)                                      \
    do {                                                        \
        cublasStatus_t status = (call);                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                  \
            std::cerr << "cuBLAS error at "                     \
                      << __FILE__ << ":" << __LINE__            \
                      << ", status = " << status << '\n';       \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)


// ============================================================
// Enumerations
// ============================================================

enum class Algorithm {
    Naive,
    StaticTiled,
    DynamicTiled,
    cuBLAS
};

enum class MemoryStrategy {
    Pageable,
    Pinned,
    PageableStreams,
    PinnedStreams,
    UnifiedPrefetch
};


const char* algorithmName(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Naive:
            return "Naive";

        case Algorithm::StaticTiled:
            return "Static_Tiled";

        case Algorithm::DynamicTiled:
            return "Dynamic_Tiled";

        case Algorithm::cuBLAS:
            return "cuBLAS";
    }

    return "Unknown";
}


const char* strategyName(MemoryStrategy strategy) {
    switch (strategy) {
        case MemoryStrategy::Pageable:
            return "Pageable";

        case MemoryStrategy::Pinned:
            return "Pinned";

        case MemoryStrategy::PageableStreams:
            return "Pageable_Streams";

        case MemoryStrategy::PinnedStreams:
            return "Pinned_Streams";

        case MemoryStrategy::UnifiedPrefetch:
            return "Unified_Prefetch";
    }

    return "Unknown";
}


// ============================================================
// Timing structures
// ============================================================

struct Timing {
    double h2d = 0.0;
    double kernel = 0.0;
    double d2h = 0.0;
    double total = 0.0;
};


struct BenchmarkResult {
    int N = 0;

    std::string strategy;
    std::string algorithm;

    double h2dMedian = 0.0;
    double kernelMedian = 0.0;
    double d2hMedian = 0.0;
    double totalMedian = 0.0;

    double totalMin = 0.0;

    bool verified = false;
};


// ============================================================
// Helpers
// ============================================================

using Clock = std::chrono::high_resolution_clock;


double elapsedMilliseconds(
        const Clock::time_point& start,
        const Clock::time_point& stop
) {
    return std::chrono::duration<double, std::milli>(
            stop - start
    ).count();
}


double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const size_t middle = values.size() / 2;

    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }

    return values[middle];
}


double minimum(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    return *std::min_element(values.begin(), values.end());
}


// ============================================================
// CUDA event timing
// ============================================================

float eventElapsed(
        cudaEvent_t start,
        cudaEvent_t stop
) {
    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
    ));

    return milliseconds;
}


// ============================================================
// Kernel launcher
// ============================================================

void launchAlgorithm(
        Algorithm algorithm,
        const BASE_TYPE* d_A,
        const BASE_TYPE* d_B,
        BASE_TYPE* d_C,
        int N,
        int rows,
        cublasHandle_t handle,
        cudaStream_t stream
) {
    const dim3 block(TILE_SIZE, TILE_SIZE);

    const dim3 grid(
            (N + TILE_SIZE - 1) / TILE_SIZE,
            (rows + TILE_SIZE - 1) / TILE_SIZE
    );

    switch (algorithm) {

        case Algorithm::Naive: {

            matrixMultNaive<<<grid, block, 0, stream>>>(
                    d_A,
                    d_B,
                    d_C,
                    N,
                    0,
                    rows
            );

            CUDA_CHECK(cudaGetLastError());

            break;
        }

        case Algorithm::StaticTiled: {

            matrixMultTiledStaticTpl<TILE_SIZE>
            <<<grid, block, 0, stream>>>(
                    d_A,
                            d_B,
                            d_C,
                            N,
                            0,
                            rows
            );

            CUDA_CHECK(cudaGetLastError());

            break;
        }

        case Algorithm::DynamicTiled: {

            const size_t sharedBytes =
                    2 * TILE_SIZE * TILE_SIZE * sizeof(BASE_TYPE);

            matrixMultTiledDynamic<<<
            grid,
            block,
            sharedBytes,
            stream
            >>>(
                    d_A,
                    d_B,
                    d_C,
                    N,
                    0,
                    rows
            );

            CUDA_CHECK(cudaGetLastError());

            break;
        }

        case Algorithm::cuBLAS: {

            // Row-major matrices are interpreted through
            // the corresponding transposed column-major form.

            // C^T = B^T * A^T

            CUBLAS_CHECK(cublasSetStream(handle, stream));

            const BASE_TYPE alpha = 1.0f;
            const BASE_TYPE beta = 0.0f;

            CUBLAS_CHECK(cublasSgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    N,
                    rows,
                    N,
                    &alpha,
                    d_B,
                    N,
                    d_A,
                    N,
                    &beta,
                    d_C,
                    N
            ));

            break;
        }
    }
}


// ============================================================
// CPU verification
// ============================================================

bool verifyAgainstReference(
        const std::vector<BASE_TYPE>& result,
        const std::vector<BASE_TYPE>& reference,
        int N
) {
    return verifyResult(result, reference, N);
}


// ============================================================
// Full-matrix benchmark
// Pageable / Pinned / Unified
// ============================================================

Timing runFullMatrix(
        Algorithm algorithm,
        MemoryStrategy strategy,
        const std::vector<BASE_TYPE>& A,
        const std::vector<BASE_TYPE>& B,
        std::vector<BASE_TYPE>& C,
        int N
) {
    Timing timing;

    const size_t bytes =
            static_cast<size_t>(N) * N * sizeof(BASE_TYPE);

    BASE_TYPE* hostA = nullptr;
    BASE_TYPE* hostB = nullptr;
    BASE_TYPE* hostC = nullptr;

    BASE_TYPE* d_A = nullptr;
    BASE_TYPE* d_B = nullptr;
    BASE_TYPE* d_C = nullptr;

    bool pinnedAllocated = false;
    bool unifiedAllocated = false;

    // --------------------------------------------------------
    // Host memory allocation
    // --------------------------------------------------------

    if (strategy == MemoryStrategy::Pinned) {

        CUDA_CHECK(cudaHostAlloc(
                &hostA,
                bytes,
                cudaHostAllocDefault
        ));

        CUDA_CHECK(cudaHostAlloc(
                &hostB,
                bytes,
                cudaHostAllocDefault
        ));

        CUDA_CHECK(cudaHostAlloc(
                &hostC,
                bytes,
                cudaHostAllocDefault
        ));

        std::copy(A.begin(), A.end(), hostA);
        std::copy(B.begin(), B.end(), hostB);

        pinnedAllocated = true;
    }
    else {
        hostA = const_cast<BASE_TYPE*>(A.data());
        hostB = const_cast<BASE_TYPE*>(B.data());
        hostC = C.data();
    }

    // --------------------------------------------------------
    // Device / Unified allocation
    // --------------------------------------------------------

    if (strategy == MemoryStrategy::UnifiedPrefetch) {

        CUDA_CHECK(cudaMallocManaged(&d_A, bytes));
        CUDA_CHECK(cudaMallocManaged(&d_B, bytes));
        CUDA_CHECK(cudaMallocManaged(&d_C, bytes));

        std::copy(A.begin(), A.end(), d_A);
        std::copy(B.begin(), B.end(), d_B);

        unifiedAllocated = true;
    }
    else {

        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
    }

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // --------------------------------------------------------
    // Warm-up
    // --------------------------------------------------------

    if (strategy == MemoryStrategy::UnifiedPrefetch) {

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        CUDA_CHECK(cudaMemPrefetchAsync(
                d_A, bytes, device
        ));

        CUDA_CHECK(cudaMemPrefetchAsync(
                d_B, bytes, device
        ));

        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else {

        CUDA_CHECK(cudaMemcpy(
                d_A, hostA, bytes, cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
                d_B, hostB, bytes, cudaMemcpyHostToDevice
        ));
    }

    launchAlgorithm(
            algorithm,
            d_A,
            d_B,
            d_C,
            N,
            N,
            handle,
            nullptr
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // H2D
    // --------------------------------------------------------

    auto totalStart = Clock::now();

    auto h2dStart = Clock::now();

    if (strategy == MemoryStrategy::UnifiedPrefetch) {

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        CUDA_CHECK(cudaMemPrefetchAsync(
                d_A, bytes, device
        ));

        CUDA_CHECK(cudaMemPrefetchAsync(
                d_B, bytes, device
        ));

        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else {

        CUDA_CHECK(cudaMemcpy(
                d_A, hostA, bytes, cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
                d_B, hostB, bytes, cudaMemcpyHostToDevice
        ));
    }

    auto h2dStop = Clock::now();

    timing.h2d = elapsedMilliseconds(
            h2dStart,
            h2dStop
    );

    // --------------------------------------------------------
    // Kernel
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventRecord(start));

    launchAlgorithm(
            algorithm,
            d_A,
            d_B,
            d_C,
            N,
            N,
            handle,
            nullptr
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    timing.kernel = eventElapsed(start, stop);

    // --------------------------------------------------------
    // D2H
    // --------------------------------------------------------

    auto d2hStart = Clock::now();

    if (strategy == MemoryStrategy::UnifiedPrefetch) {

        CUDA_CHECK(cudaMemPrefetchAsync(
                d_C,
                bytes,
                cudaCpuDeviceId
        ));

        CUDA_CHECK(cudaDeviceSynchronize());

        std::copy(d_C, d_C + N * N, C.begin());
    }
    else {

        CUDA_CHECK(cudaMemcpy(
                hostC,
                d_C,
                bytes,
                cudaMemcpyDeviceToHost
        ));
    }

    auto d2hStop = Clock::now();

    timing.d2h = elapsedMilliseconds(
            d2hStart,
            d2hStop
    );

    auto totalStop = Clock::now();

    timing.total = elapsedMilliseconds(
            totalStart,
            totalStop
    );

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (unifiedAllocated) {

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    else {

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    if (pinnedAllocated) {

        CUDA_CHECK(cudaFreeHost(hostA));
        CUDA_CHECK(cudaFreeHost(hostB));
        CUDA_CHECK(cudaFreeHost(hostC));
    }

    return timing;
}


// ============================================================
// Streaming benchmark
// Pageable + Streams / Pinned + Streams
// ============================================================

Timing runStreaming(
        Algorithm algorithm,
        MemoryStrategy strategy,
        const std::vector<BASE_TYPE>& A,
        const std::vector<BASE_TYPE>& B,
        std::vector<BASE_TYPE>& C,
        int N
) {
    Timing timing;

    const int streamsCount = std::min(NUM_STREAMS, N);

    const int rowsPerStream =
            (N + streamsCount - 1) / streamsCount;

    const size_t fullBytes =
            static_cast<size_t>(N) * N * sizeof(BASE_TYPE);

    const size_t BBytes = fullBytes;

    const bool usePinned =
            strategy == MemoryStrategy::PinnedStreams;

    BASE_TYPE* hostA = const_cast<BASE_TYPE*>(A.data());
    BASE_TYPE* hostB = const_cast<BASE_TYPE*>(B.data());
    BASE_TYPE* hostC = C.data();

    if (usePinned) {

        CUDA_CHECK(cudaHostAlloc(
                &hostA,
                fullBytes,
                cudaHostAllocDefault
        ));

        CUDA_CHECK(cudaHostAlloc(
                &hostB,
                fullBytes,
                cudaHostAllocDefault
        ));

        CUDA_CHECK(cudaHostAlloc(
                &hostC,
                fullBytes,
                cudaHostAllocDefault
        ));

        std::copy(A.begin(), A.end(), hostA);
        std::copy(B.begin(), B.end(), hostB);
    }

    BASE_TYPE* d_B = nullptr;

    CUDA_CHECK(cudaMalloc(
            &d_B,
            BBytes
    ));

    std::vector<cudaStream_t> streams(streamsCount);
    std::vector<cublasHandle_t> handles(streamsCount);

    std::vector<BASE_TYPE*> d_A(streamsCount);
    std::vector<BASE_TYPE*> d_C(streamsCount);

    std::vector<cudaEvent_t> h2dStart(streamsCount);
    std::vector<cudaEvent_t> h2dStop(streamsCount);

    std::vector<cudaEvent_t> kernelStart(streamsCount);
    std::vector<cudaEvent_t> kernelStop(streamsCount);

    std::vector<cudaEvent_t> d2hStart(streamsCount);
    std::vector<cudaEvent_t> d2hStop(streamsCount);

    std::vector<int> chunkRows(streamsCount);

    // --------------------------------------------------------
    // Allocate stream resources
    // --------------------------------------------------------

    for (int s = 0; s < streamsCount; ++s) {

        const int firstRow = s * rowsPerStream;

        const int rows = std::max(
                0,
                std::min(rowsPerStream, N - firstRow)
        );

        chunkRows[s] = rows;

        CUDA_CHECK(cudaStreamCreate(&streams[s]));

        CUBLAS_CHECK(cublasCreate(&handles[s]));

        CUBLAS_CHECK(cublasSetStream(
                handles[s],
                streams[s]
        ));

        const size_t chunkBytes =
                static_cast<size_t>(rows) * N * sizeof(BASE_TYPE);

        CUDA_CHECK(cudaMalloc(
                &d_A[s],
                chunkBytes
        ));

        CUDA_CHECK(cudaMalloc(
                &d_C[s],
                chunkBytes
        ));

        CUDA_CHECK(cudaEventCreate(&h2dStart[s]));
        CUDA_CHECK(cudaEventCreate(&h2dStop[s]));

        CUDA_CHECK(cudaEventCreate(&kernelStart[s]));
        CUDA_CHECK(cudaEventCreate(&kernelStop[s]));

        CUDA_CHECK(cudaEventCreate(&d2hStart[s]));
        CUDA_CHECK(cudaEventCreate(&d2hStop[s]));
    }

    // --------------------------------------------------------
    // Warm-up: upload B once
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
            d_B,
            hostB,
            BBytes,
            cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // Pipeline
    // --------------------------------------------------------

    auto totalStart = Clock::now();

    auto bUploadStart = Clock::now();

    CUDA_CHECK(cudaMemcpy(
            d_B,
            hostB,
            BBytes,
            cudaMemcpyHostToDevice
    ));

    auto bUploadStop = Clock::now();

    const double bUploadTime =
            elapsedMilliseconds(bUploadStart, bUploadStop);

    for (int s = 0; s < streamsCount; ++s) {

        const int firstRow = s * rowsPerStream;
        const int rows = chunkRows[s];

        if (rows == 0) {
            continue;
        }

        const size_t chunkBytes =
                static_cast<size_t>(rows) * N * sizeof(BASE_TYPE);

        const BASE_TYPE* sourceA =
                hostA + static_cast<size_t>(firstRow) * N;

        BASE_TYPE* destinationC =
                hostC + static_cast<size_t>(firstRow) * N;

        // H2D: copy A fragment.

        CUDA_CHECK(cudaEventRecord(
                h2dStart[s],
                streams[s]
        ));

        CUDA_CHECK(cudaMemcpyAsync(
                d_A[s],
                sourceA,
                chunkBytes,
                cudaMemcpyHostToDevice,
                streams[s]
        ));

        CUDA_CHECK(cudaEventRecord(
                h2dStop[s],
                streams[s]
        ));

        // Kernel.

        CUDA_CHECK(cudaEventRecord(
                kernelStart[s],
                streams[s]
        ));

        launchAlgorithm(
                algorithm,
                d_A[s],
                d_B,
                d_C[s],
                N,
                rows,
                handles[s],
                streams[s]
        );

        CUDA_CHECK(cudaEventRecord(
                kernelStop[s],
                streams[s]
        ));

        // D2H: copy C fragment.

        CUDA_CHECK(cudaEventRecord(
                d2hStart[s],
                streams[s]
        ));

        CUDA_CHECK(cudaMemcpyAsync(
                destinationC,
                d_C[s],
                chunkBytes,
                cudaMemcpyDeviceToHost,
                streams[s]
        ));

        CUDA_CHECK(cudaEventRecord(
                d2hStop[s],
                streams[s]
        ));
    }

    // Wait for all streams.

    for (int s = 0; s < streamsCount; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

    auto totalStop = Clock::now();

    timing.total = elapsedMilliseconds(
            totalStart,
            totalStop
    );

    // --------------------------------------------------------
    // Per-stream timings
    //
    // The maximum duration is used because streams overlap.
    // Summing stream times would overestimate wall-clock time.
    // --------------------------------------------------------

    float maxH2D = 0.0f;
    float maxKernel = 0.0f;
    float maxD2H = 0.0f;

    for (int s = 0; s < streamsCount; ++s) {

        if (chunkRows[s] == 0) {
            continue;
        }

        maxH2D = std::max(
                maxH2D,
                eventElapsed(h2dStart[s], h2dStop[s])
        );

        maxKernel = std::max(
                maxKernel,
                eventElapsed(kernelStart[s], kernelStop[s])
        );

        maxD2H = std::max(
                maxD2H,
                eventElapsed(d2hStart[s], d2hStop[s])
        );
    }

    timing.h2d = bUploadTime + maxH2D;
    timing.kernel = maxKernel;
    timing.d2h = maxD2H;

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    for (int s = 0; s < streamsCount; ++s) {

        CUBLAS_CHECK(cublasDestroy(handles[s]));

        CUDA_CHECK(cudaFree(d_A[s]));
        CUDA_CHECK(cudaFree(d_C[s]));

        CUDA_CHECK(cudaEventDestroy(h2dStart[s]));
        CUDA_CHECK(cudaEventDestroy(h2dStop[s]));

        CUDA_CHECK(cudaEventDestroy(kernelStart[s]));
        CUDA_CHECK(cudaEventDestroy(kernelStop[s]));

        CUDA_CHECK(cudaEventDestroy(d2hStart[s]));
        CUDA_CHECK(cudaEventDestroy(d2hStop[s]));

        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    CUDA_CHECK(cudaFree(d_B));

    if (usePinned) {

        CUDA_CHECK(cudaFreeHost(hostA));
        CUDA_CHECK(cudaFreeHost(hostB));
        CUDA_CHECK(cudaFreeHost(hostC));
    }

    return timing;
}


// ============================================================
// One benchmark case
// ============================================================

Timing runOneCase(
        Algorithm algorithm,
        MemoryStrategy strategy,
        const std::vector<BASE_TYPE>& A,
        const std::vector<BASE_TYPE>& B,
        std::vector<BASE_TYPE>& C,
        int N
) {
    if (strategy == MemoryStrategy::PageableStreams ||
        strategy == MemoryStrategy::PinnedStreams) {

        return runStreaming(
                algorithm,
                strategy,
                A,
                B,
                C,
                N
        );
    }

    return runFullMatrix(
            algorithm,
            strategy,
            A,
            B,
            C,
            N
    );
}


// ============================================================
// CSV
// ============================================================

void writeCSVHeader(std::ofstream& file) {

    file
            << "N,"
            << "MemoryStrategy,"
            << "Algorithm,"
            << "H2D_ms,"
            << "Kernel_ms,"
            << "D2H_ms,"
            << "TotalMedian_ms,"
            << "TotalMin_ms,"
            << "Verified\n";
}


void writeCSVRow(
        std::ofstream& file,
        const BenchmarkResult& result
) {
    file
            << result.N << ','
            << result.strategy << ','
            << result.algorithm << ','
            << std::fixed << std::setprecision(6)
            << result.h2dMedian << ','
            << result.kernelMedian << ','
            << result.d2hMedian << ','
            << result.totalMedian << ','
            << result.totalMin << ','
            << (result.verified ? "true" : "false")
            << '\n';
}


// ============================================================
// GPU information
// ============================================================

void printGPUInfo() {

    int device = 0;

    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties;

    CUDA_CHECK(cudaGetDeviceProperties(
            &properties,
            device
    ));

    std::cout << "\n====================================\n";
    std::cout << "GPU INFORMATION\n";
    std::cout << "====================================\n";

    std::cout << "Device: "
              << properties.name << '\n';

    std::cout << "Compute capability: "
              << properties.major << '.'
              << properties.minor << '\n';

    std::cout << "Global memory: "
              << properties.totalGlobalMem / (1024 * 1024)
              << " MB\n";

    std::cout << "Multiprocessors: "
              << properties.multiProcessorCount << '\n';

    std::cout << "Max threads per block: "
              << properties.maxThreadsPerBlock << '\n';

    std::cout << "====================================\n\n";
}


// ============================================================
// Main benchmark
// ============================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::srand(42);

    printGPUInfo();

    const std::vector<Algorithm> algorithms = {
            Algorithm::Naive,
            Algorithm::StaticTiled,
            Algorithm::DynamicTiled,
            Algorithm::cuBLAS
    };

    const std::vector<MemoryStrategy> strategies = {
            MemoryStrategy::Pageable,
            MemoryStrategy::Pinned,
            MemoryStrategy::PageableStreams,
            MemoryStrategy::PinnedStreams,
            MemoryStrategy::UnifiedPrefetch
    };

    std::ofstream csv("cuda_lab3_results.csv");

    if (!csv.is_open()) {
        std::cerr << "Cannot open CSV file.\n";
        return EXIT_FAILURE;
    }

    writeCSVHeader(csv);

    for (int N : MATRIX_SIZES) {

        std::cout << "\n====================================\n";
        std::cout << "MATRIX SIZE: " << N << " x " << N << '\n';
        std::cout << "====================================\n";

        const size_t matrixElements =
                static_cast<size_t>(N) * N;

        std::vector<BASE_TYPE> A(matrixElements);
        std::vector<BASE_TYPE> B(matrixElements);

        std::vector<BASE_TYPE> C(matrixElements);
        std::vector<BASE_TYPE> C_reference(matrixElements);

        initMatrix(A, N);
        initMatrix(B, N);

        // CPU reference is computed once for this N.
        // For larger matrices, use Naive GPU output as the
        // reference to avoid an excessively long CPU run.

        const bool useCPUReference = (N <= 1024);

        if (useCPUReference) {

            std::cout << "Computing CPU reference...\n";

            computeCPU(
                    A,
                    B,
                    C_reference,
                    N
            );
        }

        std::vector<BASE_TYPE> gpuReference;

        for (MemoryStrategy strategy : strategies) {

            std::cout << "\nMemory strategy: "
                      << strategyName(strategy)
                      << '\n';

            for (Algorithm algorithm : algorithms) {

                std::cout << "  Algorithm: "
                          << algorithmName(algorithm)
                          << " ... "
                          << std::flush;

                std::vector<double> h2dTimes;
                std::vector<double> kernelTimes;
                std::vector<double> d2hTimes;
                std::vector<double> totalTimes;

                bool verified = true;

                for (int repeat = 0;
                     repeat < NUM_WARMUP + NUM_REPEATS;
                     ++repeat) {

                    std::fill(C.begin(), C.end(), 0.0f);

                    Timing current = runOneCase(
                            algorithm,
                            strategy,
                            A,
                            B,
                            C,
                            N
                    );

                    if (repeat < NUM_WARMUP) {
                        continue;
                    }

                    h2dTimes.push_back(current.h2d);
                    kernelTimes.push_back(current.kernel);
                    d2hTimes.push_back(current.d2h);
                    totalTimes.push_back(current.total);

                    // ------------------------------------------------
                    // Verification
                    // ------------------------------------------------

                    if (useCPUReference) {

                        verified = verifyAgainstReference(
                                C,
                                C_reference,
                                N
                        );
                    }
                    else if (algorithm == Algorithm::Naive) {

                        gpuReference = C;
                    }
                    else if (!gpuReference.empty()) {

                        verified = verifyAgainstReference(
                                C,
                                gpuReference,
                                N
                        );
                    }
                }

                BenchmarkResult result;

                result.N = N;

                result.strategy = strategyName(strategy);
                result.algorithm = algorithmName(algorithm);

                result.h2dMedian = median(h2dTimes);
                result.kernelMedian = median(kernelTimes);
                result.d2hMedian = median(d2hTimes);
                result.totalMedian = median(totalTimes);

                result.totalMin = minimum(totalTimes);
                result.verified = verified;

                writeCSVRow(csv, result);
                csv.flush();

                std::cout
                        << (verified ? "OK" : "FAILED")
                        << " | H2D: " << result.h2dMedian
                        << " ms | Kernel: " << result.kernelMedian
                        << " ms | D2H: " << result.d2hMedian
                        << " ms | Total: " << result.totalMedian
                        << " ms\n";
            }
        }
    }

    csv.close();

    std::cout << "\n====================================\n";
    std::cout << "BENCHMARK FINISHED\n";
    std::cout << "Results saved to cuda_lab3_results.csv\n";
    std::cout << "====================================\n";

    return EXIT_SUCCESS;
}