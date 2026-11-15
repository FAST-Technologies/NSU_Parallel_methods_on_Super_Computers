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
__global__ void matrixMultNaive(
        const BASE_TYPE *A,
        const BASE_TYPE *B,
        BASE_TYPE *C,
        int N,
        int rowOffset,
        int rows
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int localRow = blockIdx.y * blockDim.y + threadIdx.y;
    const int globalRow = rowOffset + localRow;

    if (localRow >= rows || col >= N) {
        return;
    }

    BASE_TYPE sum = 0.0f;

    for (int k = 0; k < N; ++k) {
        sum += A[localRow * N + k] * B[k * N + col];
    }

    C[localRow * N + col] = sum;
}

// =========================================================
// 2. Блочный алгоритм (Shared Memory / Tiled)
// =========================================================
__global__ void matrixMultTiledDynamic(
        const BASE_TYPE *A,
        const BASE_TYPE *B,
        BASE_TYPE *C,
        int N,
        int rowOffset,
        int rows
) {
    // Динамическая разделяемая память (размер задается при запуске ядра)
    extern __shared__ BASE_TYPE sharedMem[];

    const int tileSize = blockDim.x;

    BASE_TYPE* shared_A = sharedMem;
    BASE_TYPE* shared_B = sharedMem + tileSize * tileSize;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * tileSize + tx;
    const int localRow = blockIdx.y * tileSize + ty;
    const int globalRow = rowOffset + localRow;

    BASE_TYPE sum = 0.0f;

    const int numTiles = (N + tileSize - 1) / tileSize;

    for (int tile = 0; tile < numTiles; ++tile) {

        const int aCol = tile * tileSize + tx;
        const int bRow = tile * tileSize + ty;

        // Load A tile.
        if (localRow < rows && aCol < N) {
            shared_A[ty * tileSize + tx] =
                    A[localRow * N + aCol];
        }
        else {
            shared_A[ty * tileSize + tx] = 0.0f;
        }

        // Load B tile.
        if (bRow < N && col < N) {
            shared_B[ty * tileSize + tx] =
                    B[bRow * N + col];
        }
        else {
            shared_B[ty * tileSize + tx] = 0.0f;
        }

        // Синхронизация блока
        __syncthreads();

        for (int k = 0; k < tileSize; ++k) {
            sum += shared_A[ty * tileSize + k] *
                   shared_B[k * tileSize + tx];
        }

        // Синхронизация перед загрузкой следующего тайла
        __syncthreads();
    }

    if (localRow < rows && col < N) {
        C[localRow * N + col] = sum;
    }
}

// ============================================================
// CPU matrix initialization
// ============================================================
void initMatrix(std::vector<BASE_TYPE>& mat, int N) {
    for (int i = 0; i < N * N; i++) {
        // mat[i] = static_cast<BASE_TYPE>(rand()) / RAND_MAX;
        // mat[i] = static_cast<BASE_TYPE>(rand() % 100) / 10.0f;
        mat[i] = static_cast<BASE_TYPE>(rand()) / (RAND_MAX / 10.0f) - 5.0f;
    }
}

// ============================================================
// Result verification
// ============================================================
bool verifyResult(const std::vector<BASE_TYPE>& C_gpu, const std::vector<BASE_TYPE>& C_cpu, int N) {
    const BASE_TYPE epsilon = 1e-3f;

    for (int i = 0; i < N * N; ++i) {

        const BASE_TYPE diff =
                std::fabs(C_gpu[i] - C_cpu[i]);

        if (diff > epsilon) {

            std::cerr
                    << "Verification failed at index " << i
                    << ": GPU = " << C_gpu[i]
                    << ", CPU = " << C_cpu[i]
                    << ", difference = " << diff
                    << '\n';

            return false;
        }
    }
    return true;
}

// ============================================================
// CPU matrix multiplication
// ============================================================
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