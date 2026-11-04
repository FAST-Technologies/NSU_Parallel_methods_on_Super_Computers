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
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    bool isBenchmark = (argc == 1);

    // Параметры по умолчанию
    int N = 1024;
    int custom_block_size = 16;

    // Наборы для режима бенчмаркинга
    const std::vector<int> benchmarkMatrixSizes = {256, 512, 1024, 2048, 4096, 8192};
    const std::vector<int> benchmarkBlockSizes = {2, 4, 8, 16, 32};

    // Парсинг аргументов для одиночного запуска: ./Lab_03 <N> [BLOCK_SIZE]
    if (!isBenchmark) {
        N = std::atoi(argv[1]);
        if (argc > 2) {
            custom_block_size = std::atoi(argv[2]);
        }
        
        // Валидация входных данных
        if (N <= 0 || custom_block_size <= 0) {
            std::cerr << "Ошибка: Неверные аргументы." << std::endl;
            std::cerr << "Использование: " << argv[0] << " [N] [BLOCK_SIZE]" << std::endl;
            std::cerr << "Пример: " << argv[0] << " 2048 32" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Инициализация cuBLAS и CUDA Events
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << std::fixed << std::setprecision(3);

    // =========================================================
    // СЦЕНАРИЙ 1: Бенчмаркинг (без аргументов)
    // =========================================================
    if (isBenchmark) {
        std::cout << "\n[РЕЖИМ БЕНЧМАРКИНГА]" << std::endl;
        
        for (int current_N : benchmarkMatrixSizes) {
            std::cout << "\n==================================================" << std::endl;
            std::cout << "Размер матрицы: " << current_N << " x " << current_N << std::endl;
            std::cout << "==================================================" << std::endl;

            size_t bytes = current_N * current_N * sizeof(BASE_TYPE);
            std::vector<BASE_TYPE> h_A(current_N * current_N), h_B(current_N * current_N);
            std::vector<BASE_TYPE> h_C_ref(current_N * current_N), h_C_gpu(current_N * current_N);

            initMatrix(h_A, current_N);
            initMatrix(h_B, current_N);

            BASE_TYPE *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, bytes));
            CUDA_CHECK(cudaMalloc(&d_B, bytes));
            CUDA_CHECK(cudaMalloc(&d_C, bytes));

            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

            // 1. Naive (эталон для GPU-верификации, блок 16x16)
            dim3 blockNaive(16, 16);
            dim3 gridNaive((current_N + 15) / 16, (current_N + 15) / 16);
            
            CUDA_CHECK(cudaEventRecord(start));
            matrixMultNaive<<<gridNaive, blockNaive>>>(d_A, d_B, d_C, current_N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float time_naive = measureTime(start, stop);
            CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));
            std::cout << "Naive (16x16):     " << time_naive << " ms" << std::endl;

            // 2. Tiled STATIC
            float best_static_time = 1e9f;
            int best_static_bs = -1;
            for (int bs : benchmarkBlockSizes) {
                dim3 blockTiled(bs, bs);
                dim3 gridTiled((current_N + bs - 1) / bs, (current_N + bs - 1) / bs);
                
                CUDA_CHECK(cudaEventRecord(start));
                if (bs == 2) {
                    matrixMultTiledStaticTpl<2><<<gridTiled, blockTiled>>>(d_A, d_B, d_C, current_N);
                } else if (bs == 4) {
                    matrixMultTiledStaticTpl<4><<<gridTiled, blockTiled>>>(d_A, d_B, d_C, current_N);
                } else if (bs == 8) {
                    matrixMultTiledStaticTpl<8><<<gridTiled, blockTiled>>>(d_A, d_B, d_C, current_N);
                } else if (bs == 16) {
                    matrixMultTiledStaticTpl<16><<<gridTiled, blockTiled>>>(d_A, d_B, d_C, current_N);
                } else if (bs == 32) {
                    matrixMultTiledStaticTpl<32><<<gridTiled, blockTiled>>>(d_A, d_B, d_C, current_N);
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float time_tiled_static = measureTime(start, stop);
                CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
                bool passed_static = verifyResult(h_C_gpu, h_C_ref, current_N);
                std::cout << "Tiled_Static (" << std::setw(2) << bs << "x" << std::setw(2) << bs << "): " 
                          << time_tiled_static << " ms  [" << (passed_static ? "PASS" : "FAIL") << "]" << std::endl;
                
                if (passed_static && time_tiled_static < best_static_time) {
                    best_static_time = time_tiled_static;
                    best_static_bs = bs;
                }
            }
            std::cout << "Tiled Static (Best): " << best_static_time << " ms (Block: " << best_static_bs << "x" << best_static_bs << ")\n";

            // 3. Tiled DYNAMIC
            float best_dynamic_time = 1e9f;
            int best_dynamic_bs = -1;

            for (int bs : benchmarkBlockSizes) {
                dim3 blockTiled(bs, bs);
                dim3 gridTiled((current_N + bs - 1) / bs, (current_N + bs - 1) / bs);
                size_t sharedMemSize = 2 * bs * bs * sizeof(BASE_TYPE);

                CUDA_CHECK(cudaEventRecord(start)); // 1. Старт таймера в потоке GPU
                matrixMultTiledDynamic<<<gridTiled, blockTiled, sharedMemSize>>>(d_A, d_B, d_C, current_N);
                CUDA_CHECK(cudaGetLastError());     // 2. Ловим ошибки конфигурации запуска
                CUDA_CHECK(cudaEventRecord(stop));  // 3. Фиксируем время окончания в потоке GPU
                CUDA_CHECK(cudaEventSynchronize(stop)); // 4. Ждем на CPU, пока GPU дойдет до метки 'stop'
                
                float time_tiled_dynamic = measureTime(start, stop);
                CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
                
                bool passed_dynamic = verifyResult(h_C_gpu, h_C_ref, current_N);
                std::cout << "Tiled_Dynamic (" << std::setw(2) << bs << "x" << std::setw(2) << bs << "): " 
                          << time_tiled_dynamic << " ms  [" << (passed_dynamic ? "PASS" : "FAIL") << "]" << std::endl;

                if (passed_dynamic && time_tiled_dynamic < best_dynamic_time) {
                    best_dynamic_time = time_tiled_dynamic;
                    best_dynamic_bs = bs;
                }
            }
            std::cout << "Tiled Dynamic (Best):" << best_dynamic_time << " ms (Block: " << best_dynamic_bs << "x" << best_dynamic_bs << ")\n";

            // 3. cuBLAS (с прогревом и правильным row-major трюком)
            const BASE_TYPE alpha = 1.0f, beta = 0.0f;
            // Warm-up (прогрев для исключения накладных расходов на инициализацию)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, current_N, current_N, current_N, 
                        &alpha, d_B, current_N, d_A, current_N, &beta, d_C, current_N);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaEventRecord(start));
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, current_N, current_N, current_N, 
                        &alpha, d_B, current_N, d_A, current_N, &beta, d_C, current_N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float time_cublas = measureTime(start, stop);
            std::cout << "cuBLAS (Warm):     " << time_cublas << " ms" << std::endl;

            // 4. CPU Верификация (только для малых N)
            if (current_N <= 1024) {
                std::vector<BASE_TYPE> h_C_cpu(current_N * current_N);
                computeCPU(h_A, h_B, h_C_cpu, current_N);
                if (verifyResult(h_C_gpu, h_C_cpu, current_N)) {
                    std::cout << "CPU Verification:  PASSED" << std::endl;
                } else {
                    std::cout << "CPU Verification:  FAILED" << std::endl;
                }
            } else {
                std::cout << "CPU Verification:  SKIPPED (N > 1024)" << std::endl;
            }

            std::cout << ">> Лучший Tiled Static:  " << best_static_time << " ms (Block: " << best_static_bs << "x" << best_static_bs << ")\n";
            std::cout << ">> Лучший Tiled Dynamic: " << best_dynamic_time << " ms (Block: " << best_dynamic_bs << "x" << best_dynamic_bs << ")\n";

            // Очистка ресурсов итерации
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
        }
    } 
    // =========================================================
    // СЦЕНАРИЙ 2: Одиночный запуск (с аргументами командной строки)
    // =========================================================
    else {
        std::cout << "\n[РЕЖИМ ОДИНОЧНОГО ЗАПУСКА]" << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << "Размер матрицы: " << N << " x " << N << std::endl;
        std::cout << "Размер блока:   " << custom_block_size << " x " << custom_block_size << std::endl;
        std::cout << "==================================================" << std::endl;

        size_t bytes = N * N * sizeof(BASE_TYPE);
        std::vector<BASE_TYPE> h_A(N * N), h_B(N * N), h_C_cpu(N * N), h_C_gpu(N * N), h_C_ref(N * N);

        initMatrix(h_A, N);
        initMatrix(h_B, N);

        BASE_TYPE *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(custom_block_size, custom_block_size);
        dim3 grid((N + custom_block_size - 1) / custom_block_size, 
                  (N + custom_block_size - 1) / custom_block_size);
        size_t sharedMemSize = 2 * custom_block_size * custom_block_size * sizeof(BASE_TYPE);

        // 1. Naive
        CUDA_CHECK(cudaEventRecord(start));
        matrixMultNaive<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_naive = measureTime(start, stop);
        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        // 2.1 Tiled Static
        CUDA_CHECK(cudaEventRecord(start));
        if (custom_block_size == 2) {
            matrixMultTiledStaticTpl<2><<<grid, block>>>(d_A, d_B, d_C, N);
        } else if (custom_block_size == 4) {
            matrixMultTiledStaticTpl<4><<<grid, block>>>(d_A, d_B, d_C, N);
        } else if (custom_block_size == 8) {
            matrixMultTiledStaticTpl<8><<<grid, block>>>(d_A, d_B, d_C, N);
        } else if (custom_block_size == 16) {
            matrixMultTiledStaticTpl<16><<<grid, block>>>(d_A, d_B, d_C, N);
        } else if (custom_block_size == 32) {
            matrixMultTiledStaticTpl<32><<<grid, block>>>(d_A, d_B, d_C, N);
        } else {
            std::cerr << "Ошибка: Неподдерживаемый размер блока для Static: " << custom_block_size << std::endl;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_tiled_static = measureTime(start, stop);
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool tiled_static_passed = verifyResult(h_C_gpu, h_C_ref, N);
        
        // 2.2 Tiled Dynamic
        CUDA_CHECK(cudaEventRecord(start));
        matrixMultTiledDynamic<<<grid, block, sharedMemSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_tiled_dynamic = measureTime(start, stop);
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool tiled_dynamic_passed = verifyResult(h_C_gpu, h_C_ref, N);

        // 3. cuBLAS
        const BASE_TYPE alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N); // Warm-up
        
        CUDA_CHECK(cudaEventRecord(start));
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_cublas = measureTime(start, stop);

        std::cout << "\nРезультаты:" << std::endl;
        std::cout << "Naive (Global):  " << time_naive << " ms" << std::endl;
        std::cout << "Tiled (Static):      " << time_tiled_static << " ms [" << (tiled_static_passed ? "PASS" : "FAIL") << "]" << std::endl;
        std::cout << "Tiled (Dynamic):     " << time_tiled_dynamic << " ms [" << (tiled_dynamic_passed ? "PASS" : "FAIL") << "]" << std::endl;
        std::cout << "cuBLAS (Warm):   " << time_cublas << " ms" << std::endl;

        if (N <= 1024) {
            computeCPU(h_A, h_B, h_C_cpu, N);
            if (verifyResult(h_C_gpu, h_C_cpu, N)) {
                std::cout << "CPU Verification:  PASSED" << std::endl;
            } else {
                std::cout << "CPU Verification:  FAILED" << std::endl;
            }
        } else {
            std::cout << "CPU Verification:  SKIPPED (N > 1024)" << std::endl;
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\nЛабораторная работа успешно завершена." << std::endl;
    return EXIT_SUCCESS;
}