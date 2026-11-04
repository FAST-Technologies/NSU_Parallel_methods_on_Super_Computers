/**
 * @file matrix_kernels.hpp.
 * @brief Лабораторная работа №3: cuBLAS Parallel Programming.
 * @details Заголовочный файл с объявлениями CUDA-ядер и вспомогательных функций
 *        для умножения матриц.
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 04.11.2026.
 * @version 1.0.
 */

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <vector>
#include <cmath>
#include <iostream>
#include <cstdlib>

/**
 * @brief Размер блока (тайла) для оптимизированного алгоритма.
 * @note Можно изменять на 16 или 32 для проведения экспериментов.
 */
#define BLOCK_SIZE 16

/**
 * @brief Базовый тип данных для элементов матрицы.
 */
using BASE_TYPE = float;

/**
 * @brief Макрос для проверки кодов возврата функций CUDA.
 * @details Если функция возвращает ошибку, макрос выводит сообщение
 *          с именем файла и номером строки, после чего завершает программу.
 * @param call Вызываемая функция CUDA.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifdef __CUDACC__

/**
 * @brief Наивный алгоритм умножения матриц (использует только глобальную память).
 * @details Каждый поток вычисляет один элемент результирующей матрицы C,
 *          считывая данные напрямую из глобальной памяти видеокарты.
 * @param A Указатель на матрицу A в глобальной памяти (размер N x N).
 * @param B Указатель на матрицу B в глобальной памяти (размер N x N).
 * @param C Указатель на результирующую матрицу C в глобальной памяти.
 * @param N Размерность квадратных матриц.
 */
__global__ void matrixMultNaive(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int N);

/**
 * @brief Блочный (Tiled) алгоритм умножения матриц с использованием СТАТИЧЕСКОЙ разделяемой памяти.
 * @details Матрицы разбиваются на блоки размером BLOCK_SIZE x BLOCK_SIZE.
 *          Потоки кооперативно загружают фрагменты в быструю разделяемую память (Shared Memory),
 *          синхронизируются и выполняют вычисления, многократно используя загруженные данные.
 * @param A Указатель на матрицу A в глобальной памяти.
 * @param B Указатель на матрицу B в глобальной памяти.
 * @param C Указатель на результирующую матрицу C в глобальной памяти.
 * @param N Размерность квадратных матриц.
 */
template <int TILE_SIZE>
__global__ void matrixMultTiledStaticTpl(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int N);


/**
 * @brief Блочный (Tiled) алгоритм умножения матриц с использованием ДИНАМИЧЕСКОЙ разделяемой памяти.
 * @details Матрицы разбиваются на блоки размером BLOCK_SIZE x BLOCK_SIZE.
 *          Потоки кооперативно загружают фрагменты в быструю разделяемую память (Shared Memory),
 *          синхронизируются и выполняют вычисления, многократно используя загруженные данные.
 * @param A Указатель на матрицу A в глобальной памяти.
 * @param B Указатель на матрицу B в глобальной памяти.
 * @param C Указатель на результирующую матрицу C в глобальной памяти.
 * @param N Размерность квадратных матриц.
 */
__global__ void matrixMultTiledDynamic(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int N);

/**
 * @brief Измерение времени выполнения между двумя CUDA-событиями.
 * @details Вычисляет прошедшее время (elapsed time) в миллисекундах между 
 *          событиями start и stop. 
 * @warning Функция предполагает, что события уже были синхронизированы 
 *          (например, через cudaEventSynchronize(stop)) перед её вызовом, 
 *          иначе результат может быть некорректным (равным 0).
 * @param start Событие, отмечающее начало измеряемого интервала.
 * @param stop  Событие, отмечающее конец измеряемого интервала.
 * @return Время выполнения в миллисекундах (float).
 */
inline float measureTime(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, start, stop);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Event Error: " << cudaGetErrorString(err) 
                  << " при измерении времени." << std::endl;
    }
    return ms;
}

#endif // __CUDACC__

/**
 * @brief Инициализация матрицы случайными значениями.
 * @param[out] mat Ссылка на вектор, представляющий матрицу (размер N x N).
 * @param N Размерность матрицы.
 */
void initMatrix(std::vector<BASE_TYPE>& mat, int N);

/**
 * @brief Проверка корректности результатов умножения матриц.
 * @param C_gpu Результаты, полученные на GPU.
 * @param C_cpu Эталонные результаты, полученные на CPU.
 * @param N Размерность матриц.
 * @return true, если максимальная поэлементная разница меньше допуска (1e-2), иначе false.
 */
bool verifyResult(const std::vector<BASE_TYPE>& C_gpu, const std::vector<BASE_TYPE>& C_cpu, int N);

/**
 * @brief Последовательное (CPU) умножение матриц для верификации.
 * @param A Исходная матрица A (размер N x N).
 * @param B Исходная матрица B (размер N x N).
 * @param C Выходная матрица C (размер N x N).
 * @param N Размерность матриц.
 */
void computeCPU(const std::vector<BASE_TYPE>& A, const std::vector<BASE_TYPE>& B, std::vector<BASE_TYPE>& C, int N);