/**
 * @file matrix_kernels.cu.
 * @brief Лабораторная работа №3: cuBLAS Parallel Programming.
 * @details Реализация CUDA-ядер и вспомогательных функций для умножения матриц.
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 04.11.2026.
 * @version 1.0.
 */

#include "matrix_kernels.hpp"

// =========================================================
// 1. Наивный алгоритм (Global Memory)
// =========================================================
__global__ void matrixMultNaive(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        BASE_TYPE sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// =========================================================
// 2. Блочный алгоритм (Shared Memory / Tiled)
// =========================================================
__global__ void matrixMultTiled(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int N) {
    __shared__ BASE_TYPE shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    BASE_TYPE sum = 0.0f;

    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        int a_col = tile * BLOCK_SIZE + threadIdx.x;
        int b_row = tile * BLOCK_SIZE + threadIdx.y;

        // Загрузка в разделяемую память с проверкой границ
        shared_A[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        shared_B[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        // Синхронизация блока
        __syncthreads();

        // Вычисление частичной суммы
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        // Синхронизация перед загрузкой следующего тайла
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// =========================================================
// Вспомогательные функции (Host)
// =========================================================
void initMatrix(std::vector<BASE_TYPE>& mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<BASE_TYPE>(rand()) / RAND_MAX;
    }
}

bool verifyResult(const std::vector<BASE_TYPE>& C_gpu, const std::vector<BASE_TYPE>& C_cpu, int N) {
    const float epsilon = 1e-2f;
    for (int i = 0; i < N * N; i++) {
        if (std::abs(C_gpu[i] - C_cpu[i]) > epsilon) {
            std::cerr << "Ошибка верификации на индексе " << i
                      << " (CPU: " << C_cpu[i] << ", GPU: " << C_gpu[i] << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void computeCPU(const std::vector<BASE_TYPE>& A, const std::vector<BASE_TYPE>& B, std::vector<BASE_TYPE>& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BASE_TYPE sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}