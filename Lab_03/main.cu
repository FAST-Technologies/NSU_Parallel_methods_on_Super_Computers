/**
 * @file main.cu
 * @brief Лабораторная работа №3: cuBLAS Parallel Programming.
 * @details Точка входа в программу. Инициализация, запуск ядер, замеры времени и верификация.
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 04.11.2026.
 * @version 1.0.
 */

#include "matrix_kernels.hpp"
#include <chrono>

int main(int argc, char** argv) {
    int N = 1024; // Размер матрицы по умолчанию
    int custom_block_size = BLOCK_SIZE;

    // Парсинг аргументов: ./Lab_03 <N> [BLOCK_SIZE]
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) custom_block_size = std::atoi(argv[2]);

    dim3 threadsPerBlock(custom_block_size, custom_block_size);
    dim3 blocksPerGrid((N + custom_block_size - 1) / custom_block_size,
                       (N + custom_block_size - 1) / custom_block_size)

    std::cout << "==================================================" << std::endl;
    std::cout << "Размер матрицы: " << N << "x" << N << std::endl;
    std::cout << "Размер блока:   " << custom_block_size << "x" << custom_block_size << std::endl;
    std::cout << "==================================================" << std::endl;

    size_t bytes = N * N * sizeof(BASE_TYPE);
    std::vector<BASE_TYPE> h_A(N * N), h_B(N * N), h_C_cpu(N * N), h_C_gpu(N * N);

    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Выделение памяти на GPU
    BASE_TYPE *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_naive = 0, time_tiled = 0, time_cublas = 0;

    // -----------------------------------------------------
    // 1. Наивный алгоритм
    // -----------------------------------------------------
    cudaEventRecord(start);
    matrixMultNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_naive, start, stop);

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // -----------------------------------------------------
    // 2. Блочный алгоритм (Shared Memory)
    // -----------------------------------------------------
    cudaEventRecord(start);
    matrixMultTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tiled, start, stop);

    // -----------------------------------------------------
    // 3. cuBLAS (с прогревом)
    // -----------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warm-up (прогрев для исключения накладных расходов на инициализацию)
    const BASE_TYPE alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    // Замер
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cublas, start, stop);

    cublasDestroy(handle);

    // -----------------------------------------------------
    // Верификация и вывод
    // -----------------------------------------------------
    if (N <= 512) { // Проверка только для небольших матриц, чтобы не ждать час
        computeCPU(h_A, h_B, h_C_cpu, N);
        if (verifyResult(h_C_gpu, h_C_cpu, N)) {
            std::cout << "Верификация: PASSED" << std::endl;
        } else {
            std::cout << "Верификация: FAILED" << std::endl;
        }
    } else {
        std::cout << "Верификация: SKIPPED (N > 512)" << std::endl;
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Naive (Global):  " << time_naive << " ms" << std::endl;
    std::cout << "Tiled (Shared):  " << time_tiled << " ms" << std::endl;
    std::cout << "cuBLAS (Warm):   " << time_cublas << " ms" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Очистка ресурсов
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}