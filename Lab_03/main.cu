
/**
 * @file main.cu
 * @brief Точка входа лабораторной работы №3.
 *
 * @details Поддерживает два режима:
 *          1. Полный бенчмарк без аргументов командной строки.
 *          2. Одиночный запуск для указанного размера матрицы.
 *
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 15.11.2026.
 * @version 2.1.
 */

#include "config.hpp"
#include "matrix_kernels.hpp"
#include "benchmark.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

/** Максимальный размер матрицы в режиме одиночного запуска. */
constexpr int MAX_SINGLE_MATRIX_SIZE = 16384;

/** Название точности для CSV. */
constexpr const char* kPrecisionName =
    std::is_same_v<BASE_TYPE, float> ? "f32" : "f64";

/**
 * @brief Преобразует строку в положительное целое число.
 *
 * @param text Исходная строка.
 * @param value Переменная для сохранения результата.
 * @return true, если строка содержит корректное положительное число.
 */
bool parsePositiveInt(std::string_view text, int& value) {
    if (text.empty()) {
        return false;
    }

    int parsedValue = 0;

    const auto result = std::from_chars(
        text.data(),
        text.data() + text.size(),
        parsedValue
    );

    if (result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() ||
        parsedValue <= 0) {
        return false;
    }

    value = parsedValue;
    return true;
}

/**
 * @brief Выводит справку по использованию программы.
 *
 * @param executable Имя исполняемого файла.
 */
void printUsage(const char* executable) {
    std::cout
    << "\nCUDA Matrix Multiplication Benchmark\n\n"
    << "Usage:\n"
    << " " << executable << "\n"
    << " Full benchmark over MATRIX_SIZES, default block size (" << TILE_SIZE << ").\n\n"
    << " " << executable << " N [BLOCK_SIZE]\n"
    << " Single run for matrix size N.\n\n"
    << " " << executable << " all [BLOCK_SIZE]\n"
    << " Full benchmark over MATRIX_SIZES.\n\n"
    << " " << executable << " --help\n"
    << " Show this help message.\n\n"
    << "Constraints:\n"
    << " 1 <= N <= " << MAX_SINGLE_MATRIX_SIZE << "\n"
    << " BLOCK_SIZE in {2, 4, 8, 16, 32}, default " << TILE_SIZE << ".\n\n"
    << "Examples:\n"
    << " " << executable << " 2048 16\n"
    << " " << executable << " all 32\n";
}

/**
 * @brief Разбирает аргументы командной строки.
 *
 * @param argc Количество аргументов.
 * @param argv Массив аргументов.
 * @param matrixSizes Набор размеров матриц для запуска.
 * @param singleRun Признак одиночного режима.
 * @return true, если аргументы корректны и запуск можно продолжать.
 */
bool parseArguments(
    int argc,
    char** argv,
    std::vector<int>& matrixSizes,
    bool& singleRun,
    int& blockSize
) {
    singleRun = false;
    blockSize = TILE_SIZE;
    matrixSizes.assign(
        std::begin(MATRIX_SIZES),
        std::end(MATRIX_SIZES)
    );

    if (argc == 1) {
        return true;
    }

    const std::string_view first = argv[1];

    if (argc == 2 && (first == "--help" || first == "-h")) {
        printUsage(argv[0]);
        return false;
    }

    if (argc > 3) {
        std::cerr << "Error: invalid number of arguments.\n";
        printUsage(argv[0]);
        return false;
    }

    if (first != "all") {
        int N = 0;

        if (!parsePositiveInt(first, N)) {
            std::cerr << "Error: N must be a positive integer or 'all'.\n";
            printUsage(argv[0]);
            return false;
        }

        if (N > MAX_SINGLE_MATRIX_SIZE) {
            std::cerr << "Error: N must not exceed "
                    << MAX_SINGLE_MATRIX_SIZE << ".\n";
            return false;
        }

        singleRun = true;
        matrixSizes.assign(1, N);
    }

    if (argc == 3) {
        int requested = 0;

        if (!parsePositiveInt(argv[2], requested) ||
            !isSupportedBlockSize(requested)) {
            std::cerr << "Error: BLOCK_SIZE must be one of 2, 4, 8, 16, 32.\n";
            return false;
        }

        blockSize = requested;
    }

    return true;
}

} // namespace

/**
 * @brief Точка входа лабораторной работы №3.
 *
 * Выполняет серию экспериментов по сравнению алгоритмов
 * умножения квадратных матриц и стратегий передачи данных
 * между оперативной памятью CPU и памятью GPU.
 *
 * Поддерживает полный бенчмарк и одиночный запуск с заданным N.
 *
 * Для каждого размера матриц:
 * 1. Создаёт и инициализирует входные матрицы A и B.
 * 2. Вычисляет эталонный результат на CPU для небольших матриц.
 * 3. Выполняет прогревочные и измерительные запуски алгоритмов.
 * 4. Проверяет вычисленный результат.
 * 5. Вычисляет медианное и минимальное время выполнения.
 * 6. Сохраняет результаты экспериментов в CSV-файл.
 *
 * Для N > 1024 полный CPU-эталон не вычисляется: используется verifySampled
 * (выборочные элементы и полные строки на границах чанков стримов).
 *
 * @return EXIT_SUCCESS при успешном завершении;
 *         EXIT_FAILURE при ошибке аргументов или CSV-файла.
 */
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::vector<int> matrixSizes;
    bool singleRun = false;
    int blockSize = TILE_SIZE;

    if (!parseArguments(argc, argv, matrixSizes, singleRun, blockSize)) {
        return (argc == 2 &&
                (std::string_view(argv[1]) == "--help" ||
                 std::string_view(argv[1]) == "-h"))
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
    }
    setBlockSize(blockSize);
    std::srand(42);

    std::cout << std::fixed << std::setprecision(6);

    if (singleRun) {
        std::cout << "\n[SINGLE-RUN MODE]\n";
    } else {
        std::cout << "\n[FULL BENCHMARK MODE]\n";
    }

    printGPUInfo();
    const GPUInfo gpuInfo = getGPUInfo();

    std::cout << "Block size: " << blockSize << " x " << blockSize << '\n';
    std::cout << "Using block size: " << getBlockSize() << '\n';

    // Алгоритмы умножения матриц, участвующие в сравнении.
    const std::vector<Algorithm> algorithms = {
        Algorithm::Naive,
        Algorithm::StaticTiled,
        Algorithm::DynamicTiled,
        Algorithm::cuBLAS
    };

    // Базовые стратегии размещения памяти и потоковой передачи данных.
    std::vector<MemoryStrategy> strategies = {
        MemoryStrategy::Pageable,
        MemoryStrategy::Pinned,
        MemoryStrategy::PageableStreams,
        MemoryStrategy::PinnedStreams
    };

    // Unified Memory включается с учётом платформы и возможностей GPU.
#ifdef __linux__
    if (isUnifiedPrefetchSupported()) {
        strategies.push_back(MemoryStrategy::UnifiedPrefetch);
    } else {
        std::cout
            << "Skipping Unified Prefetch: "
            << "GPU does not support concurrent managed access.\n";
    }
    strategies.push_back(MemoryStrategy::UnifiedManaged);
#elif defined(_WIN32)
    strategies.push_back(MemoryStrategy::UnifiedManaged);

    std::cout
        << "Note: Windows Unified Memory has limited support "
        << "(concurrentManagedAccess = 0). "
        << "Performance may be significantly lower.\n";
#endif

    // Раздельные файлы для полного и одиночного режимов.
    const std::string csvFileName =
        std::string(singleRun ? "cuda_lab3_single" : "cuda_lab3") +
        (std::is_same_v<BASE_TYPE, float> ? "_f32" : "_f64") +
        "_b" + std::to_string(getBlockSize()) +
        (singleRun ? "_N" + std::to_string(matrixSizes.front()) : "") +
        ".csv";

    std::ofstream csv(csvFileName, std::ios::out | std::ios::trunc);

    if (!csv.is_open()) {
        std::cerr << "Cannot open CSV file: "
                  << csvFileName << '\n';
        return EXIT_FAILURE;
    }

    writeCSVHeader(csv);

    const auto printMetric = [](double value) {
        if (value > 0.0) {
            std::cout << value;
        } else {
            std::cout << "N/A";
        }
    };

    for (int N : matrixSizes) {
        std::cout << "\n====================================\n";
        std::cout << "MATRIX SIZE: " << N << " x " << N << '\n';
        std::cout << "====================================\n";

        const std::size_t matrixElements =
            static_cast<std::size_t>(N) * static_cast<std::size_t>(N);

        std::vector<BASE_TYPE> A(matrixElements);
        std::vector<BASE_TYPE> B(matrixElements);
        std::vector<BASE_TYPE> C(matrixElements);
        std::vector<BASE_TYPE> C_reference;

        initMatrix(A, N);
        initMatrix(B, N);

        // CPU-эталон вычисляется только для небольших матриц.
        const bool useCPUReference = (N <= 1024);

        if (useCPUReference) {
            std::cout << "Computing CPU reference...\n";
            computeCPU(A, B, C_reference, N);
        } else {
            std::cout
                << "CPU full reference skipped (N > 1024).\n"
                << "Independent sampled CPU verification will be used.\n";
        }

        std::vector<BASE_TYPE> C_cpu;

        std::cout << "\nBenchmarking CPU baseline...\n";

        const double cpuBaselineMs =
            benchmarkCPU(A, B, C_cpu, N);

        std::cout << "CPU baseline: "
                  << cpuBaselineMs << " ms\n";

        for (MemoryStrategy strategy : strategies) {
            std::cout << "\nMemory strategy: "
                      << strategyName(strategy)
                      << '\n';

            std::vector<BenchmarkResult> strategyResults;
            strategyResults.reserve(algorithms.size());

            for (Algorithm algorithm : algorithms) {
                std::cout << "  Algorithm: "
                          << algorithmName(algorithm)
                          << " ... "
                          << std::flush;

                std::vector<double> h2dTimes;
                std::vector<double> kernelTimes;
                std::vector<double> d2hTimes;
                std::vector<double> totalTimes;

                h2dTimes.reserve(NUM_REPEATS);
                kernelTimes.reserve(NUM_REPEATS);
                d2hTimes.reserve(NUM_REPEATS);
                totalTimes.reserve(NUM_REPEATS);

                // Прогревочные запуски не включаются в статистику.
                for (int repeat = 0;
                     repeat < NUM_WARMUP + NUM_REPEATS;
                     ++repeat) {

                    std::fill(
                        C.begin(),
                        C.end(),
                        BASE_TYPE{0}
                    );

                    const Timing current = runOneCase(
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
                }

                bool verified = false;
                // Проверка результата после последнего измерительного запуска.
                if (useCPUReference) {
                    verified = verifyResult(
                        C,
                        C_reference,
                        N
                    );
                } else {
                    verified = verifySampled(
                        C,
                        A,
                        B,
                        N
                    );
                }

                BenchmarkResult result;

                result.N = N;
                result.blockSize = getBlockSize();
                result.precision = kPrecisionName;
                result.strategy = strategyName(strategy);
                result.algorithm = algorithmName(algorithm);

                result.verified = verified;
                result.cpuMs = cpuBaselineMs;

                result.h2dMedian = median(h2dTimes);
                result.kernelMedian = median(kernelTimes);
                result.d2hMedian = median(d2hTimes);
                result.totalMedian = median(totalTimes);

                result.totalMin = minimum(totalTimes);
                strategyResults.push_back(result);

                std::cout
                    << (verified ? "OK" : "FAILED")
                    << " | H2D: " << result.h2dMedian
                    << " ms | Kernel: " << result.kernelMedian
                    << " ms | D2H: " << result.d2hMedian
                    << " ms | Total: " << result.totalMedian
                    << " ms\n";
            }
            // -----------------------------------------------------
            // 4. Расчёт производных метрик.
            // -----------------------------------------------------

            calculateBenchmarkMetrics(
                strategyResults,
                gpuInfo
            );

            // -----------------------------------------------------
            // 5. Запись CSV и расширенный вывод.
            // -----------------------------------------------------

            for (const BenchmarkResult& result : strategyResults) {
                writeCSVRow(csv, result);
                csv.flush();

                std::cout
                    << "\n  [" << result.algorithm << "] "
                    << (result.verified ? "VERIFIED" : "NOT VERIFIED")
                    << '\n';

                std::cout
                    << "    H2D: "
                    << result.h2dMedian << " ms"
                    << " | Equivalent: "
                    << result.equivalentGflopsH2D
                    << " GFLOPS\n";

                std::cout
                    << "    Kernel: "
                    << result.kernelMedian << " ms"
                    << " | GFLOPS: "
                    << result.gflopsKernel
                    << " | TFLOPS: "
                    << result.tflopsKernel
                    << '\n';

                std::cout
                    << "    D2H: "
                    << result.d2hMedian << " ms"
                    << " | Equivalent: "
                    << result.equivalentGflopsD2H
                    << " GFLOPS\n";

                std::cout
                    << "    Total: "
                    << result.totalMedian << " ms"
                    << " | Min: "
                    << result.totalMin << " ms"
                    << " | GFLOPS: "
                    << result.gflopsTotal
                    << " | TFLOPS: "
                    << result.tflopsTotal
                    << '\n';

                std::cout
                    << "    CPU baseline: "
                    << result.cpuMs << " ms\n";

                std::cout << "    CPU speedup (Kernel): ";
                printMetric(result.speedupCPUKernel);
                std::cout << "x | CPU speedup (Total): ";
                printMetric(result.speedupCPUTotal);
                std::cout << "x\n";

                std::cout << "    Naive speedup (Kernel): ";
                printMetric(result.speedupNaiveKernel);
                std::cout << "x | Naive speedup (Total): ";
                printMetric(result.speedupNaiveTotal);
                std::cout << "x\n";

                std::cout << "    Relative efficiency (Kernel): ";
                printMetric(result.efficiencyKernelPct);
                std::cout << "% | Relative efficiency (Total): ";
                printMetric(result.efficiencyTotalPct);
                std::cout << "%\n";

                if (result.peakUtilizationKernelPct > 0.0 ||
                    result.peakUtilizationTotalPct > 0.0) {

                    std::cout << "    Peak utilization (Kernel): ";
                    printMetric(result.peakUtilizationKernelPct);
                    std::cout << "% | Peak utilization (Total): ";
                    printMetric(result.peakUtilizationTotalPct);
                    std::cout << "%\n";
                }
            }
        }
    }

    csv.close();

    std::cout << "\n====================================\n";
    std::cout << "BENCHMARK FINISHED\n";
    std::cout << "Results saved to " << csvFileName << '\n';
    std::cout << "====================================\n";

    return EXIT_SUCCESS;
}
