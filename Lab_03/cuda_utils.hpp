
/**
 * @file cuda_utils.hpp
 * @brief Вспомогательные средства обработки ошибок CUDA.
 *
 * @details Предоставляет макрос CUDA_CHECK для проверки результатов
 *          вызова функций CUDA Runtime API.
 *
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 15.11.2026.
 * @version 2.0.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

/**
 * @brief Макрос для проверки кода возврата функции CUDA Runtime API.
 *
 * @details Выполняет переданный вызов CUDA-функции и проверяет
 *          полученный код ошибки. При возникновении ошибки выводит
 *          её описание, имя исходного файла и номер строки,
 *          после чего завершает выполнение программы.
 *
 * @param call Вызов функции CUDA Runtime API, возвращающей cudaError_t.
 *
 * @note Макрос предназначен для проверки синхронных кодов возврата.
 *       Для обнаружения ошибок асинхронного выполнения необходимо
 *       дополнительно использовать синхронизацию CUDA.
 */
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        const cudaError_t err = (call);                       \
        if (err != cudaSuccess) {                             \
            std::cerr                                        \
                << "CUDA Error: "                            \
                << cudaGetErrorString(err)                   \
                << " at " << __FILE__ << ":" << __LINE__     \
                << std::endl;                                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)
