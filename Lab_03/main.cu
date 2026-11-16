
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
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

/** Максимальный размер матрицы в режиме одиночного запуска. */
constexpr int MAX_SINGLE_MATRIX_SIZE = 16384;

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
        << "  " << executable << "\n"
        << "      Run the full benchmark using MATRIX_SIZES.\n\n"
        << "  " << executable << " N\n"
        << "      Run the benchmark for one matrix size N.\n\n"
        << "  " << executable << " N BLOCK_SIZE\n"
        << "      Run with an explicitly specified block size.\n"
        << "      Currently supported BLOCK_SIZE: " << TILE_SIZE << ".\n\n"
        << "  " << executable << " --help\n"
        << "      Show this help message.\n\n"
        << "Constraints:\n"
        << "  1 <= N <= " << MAX_SINGLE_MATRIX_SIZE << "\n"
        << "  BLOCK_SIZE must equal TILE_SIZE (" << TILE_SIZE << ").\n\n"
        << "Examples:\n"
        << "  " << executable << "\n"
        << "  " << executable << " 2048\n"
        << "  " << executable << " 2048 " << TILE_SIZE << "\n";
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
    bool& singleRun
) {
    if (argc == 1) {
        singleRun = false;

        matrixSizes.assign(
            std::begin(MATRIX_SIZES),
            std::end(MATRIX_SIZES)
        );

        return true;
    }

    if (argc == 2 &&
        (std::string_view(argv[1]) == "--help" ||
         std::string_view(argv[1]) == "-h")) {
        printUsage(argv[0]);
        return false;
    }

    if (argc < 2 || argc > 3) {
        std::cerr << "Error: invalid number of arguments.\n";
        printUsage(argv[0]);
        return false;
    }

    int N = 0;

    if (!parsePositiveInt(argv[1], N)) {
        std::cerr << "Error: N must be a positive integer.\n";
        printUsage(argv[0]);
        return false;
    }

    if (N > MAX_SINGLE_MATRIX_SIZE) {
        std::cerr
            << "Error: N must not exceed "
            << MAX_SINGLE_MATRIX_SIZE << ".\n";
        return false;
    }

    if (argc == 3) {
        int blockSize = 0;

        if (!parsePositiveInt(argv[2], blockSize)) {
            std::cerr << "Error: BLOCK_SIZE must be a positive integer.\n";
            return false;
        }

        if (blockSize != TILE_SIZE) {
            std::cerr
                << "Error: this version supports only BLOCK_SIZE = "
                << TILE_SIZE << ".\n"
                << "The CUDA kernels currently use the compile-time "
                << "TILE_SIZE configuration.\n";
            return false;
        }
    }

    singleRun = true;
    matrixSizes.push_back(N);

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
 * Для больших матриц результат наивного CUDA-алгоритма
 * используется как эталон для сравнения остальных GPU-алгоритмов.
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

    if (!parseArguments(argc, argv, matrixSizes, singleRun)) {
        // --help также завершает работу через этот путь.
        return (argc == 2 &&
                (std::string_view(argv[1]) == "--help" ||
                 std::string_view(argv[1]) == "-h"))
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
    }

    std::srand(42);

    std::cout << std::fixed << std::setprecision(6);

    if (singleRun) {
        std::cout << "\n[SINGLE-RUN MODE]\n";
    } else {
        std::cout << "\n[FULL BENCHMARK MODE]\n";
    }

    printGPUInfo();

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
#elif defined(_WIN32)
    strategies.push_back(MemoryStrategy::UnifiedManaged);
#endif

    // Раздельные файлы для полного и одиночного режимов.
    const char* csvFileName = singleRun
        ? "cuda_lab3_single_results.csv"
        : "cuda_lab3_results.csv";

    std::ofstream csv(csvFileName);

    if (!csv.is_open()) {
        std::cerr << "Cannot open CSV file: "
                  << csvFileName << '\n';
        return EXIT_FAILURE;
    }

    writeCSVHeader(csv);

    for (int N : matrixSizes) {
        std::cout << "\n====================================\n";
        std::cout << "MATRIX SIZE: " << N << " x " << N << '\n';
        std::cout << "====================================\n";

        const std::size_t matrixElements =
            static_cast<std::size_t>(N) * static_cast<std::size_t>(N);

        std::vector<BASE_TYPE> A(matrixElements);
        std::vector<BASE_TYPE> B(matrixElements);
        std::vector<BASE_TYPE> C(matrixElements);
        std::vector<BASE_TYPE> C_reference(matrixElements);

        initMatrix(A, N);
        initMatrix(B, N);

        // CPU-эталон вычисляется только для небольших матриц.
        const bool useCPUReference = (N <= 1024);

        if (useCPUReference) {
            std::cout << "Computing CPU reference...\n";

            computeCPU(A, B, C_reference, N);
        } else {
            std::cout
                << "CPU reference skipped (N > 1024).\n"
                << "Naive CUDA result will be used for GPU comparison.\n";
        }

        // Эталон GPU обновляется при запуске Naive
        // для каждой стратегии памяти.
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

                // Проверка результата после последнего измерительного запуска.
                if (useCPUReference) {
                    verified = verifyResult(
                        C,
                        C_reference,
                        N
                    );
                } else if (algorithm == Algorithm::Naive) {
                    gpuReference = C;
                    verified = true;
                } else if (!gpuReference.empty()) {
                    verified = verifyResult(
                        C,
                        gpuReference,
                        N
                    );
                } else {
                    verified = false;
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
    std::cout << "Results saved to " << csvFileName << '\n';
    std::cout << "====================================\n";

    return EXIT_SUCCESS;
}
