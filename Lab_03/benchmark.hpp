
/**
 * @file benchmark.hpp
 * @brief Интерфейс подсистемы бенчмаркинга CUDA.
 *
 * @details Содержит перечисления алгоритмов и стратегий памяти,
 *          структуры результатов измерений, а также объявления
 *          функций для запуска экспериментов, обработки статистики
 *          и сохранения результатов в CSV-файл.
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 15.11.2026.
 * @version 2.0.
 */

#pragma once

#include "config.hpp"

#include <fstream>
#include <string>
#include <vector>

/**
 * @brief Вычислительный алгоритм умножения матриц.
 */
enum class Algorithm {
    Naive,         ///< Наивное умножение с использованием глобальной памяти.
    StaticTiled,   ///< Блочное умножение со статической shared memory.
    DynamicTiled,  ///< Блочное умножение с динамической shared memory.
    cuBLAS         ///< Умножение с использованием библиотеки NVIDIA cuBLAS.
};

/**
 * @brief Стратегия размещения и передачи данных между CPU и GPU.
 */
enum class MemoryStrategy {
    Pageable,         ///< Обычная страничная память CPU и полная передача матриц.
    Pinned,           ///< Закреплённая страничная память CPU и полная передача матриц.
    PageableStreams,  ///< Обычная память CPU и передача частей матриц через CUDA streams.
    PinnedStreams,    ///< Закреплённая память CPU и передача частей матриц через CUDA streams.
    UnifiedPrefetch,  ///< Unified Memory с явным предварительным перемещением данных.
    UnifiedManaged    ///< Unified Memory с автоматической миграцией страниц.
};

/**
 * @brief Временные характеристики одного запуска бенчмарка.
 *
 * @details Все значения времени представлены в миллисекундах.
 *
 *          Для потоковых стратегий H2D, Kernel и D2H представляют
 *          максимальные длительности соответствующих операций
 *          среди CUDA streams, а не последовательные составляющие Total.
 *
 *          Total измеряется отдельно и включает полную стоимость
 *          одного запуска: подготовку host-буферов, создание CUDA-ресурсов,
 *          выделение памяти, передачу данных, вычисления и освобождение
 *          ресурсов.
 *
 *          Для Unified Managed автоматическая миграция страниц может
 *          учитываться во времени Kernel и Total
 */
struct Timing {
    double h2d = 0.0;     ///< Время передачи данных с CPU на GPU.
    double kernel = 0.0;  ///< Время выполнения вычислений на GPU.
    double d2h = 0.0;     ///< Время получения результата с GPU на CPU.
    double total = 0.0;   ///< Полное измеренное время одного запуска.
};

/**
 * @brief Итоговые статистические данные одного эксперимента.
 *
 * @details Содержит размер матриц, названия использованного алгоритма
 *          и стратегии памяти, медианные значения времени выполнения,
 *          минимальное полное время и результат проверки корректности.
 */
struct BenchmarkResult {
    int N = 0;                  ///< Размер квадратных матриц N x N.
    int blockSize = 0;           ///< Размер стороны CUDA-блока (тайла).

    std::string strategy;       ///< Название стратегии управления памятью.
    std::string algorithm;      ///< Название вычислительного алгоритма.

    double h2dMedian = 0.0;     ///< Медианное время передачи H2D, мс.
    double kernelMedian = 0.0;  ///< Медианное время выполнения ядра, мс.
    double d2hMedian = 0.0;     ///< Медианное время передачи D2H, мс.
    double totalMedian = 0.0;   ///< Медианное полное время выполнения, мс.

    double totalMin = 0.0;      ///< Минимальное полное время выполнения, мс.

    bool verified = false;      ///< Результат проверки корректности вычислений.
};

/**
 * @brief Возвращает текстовое имя вычислительного алгоритма.
 * @param algorithm Выбранный алгоритм.
 * @return Строковое представление алгоритма.
 */
const char* algorithmName(Algorithm algorithm);

/**
 * @brief Возвращает текстовое имя стратегии управления памятью.
 * @param strategy Выбранная стратегия памяти.
 * @return Строковое представление стратегии.
 */
const char* strategyName(MemoryStrategy strategy);

/**
 * @brief Выводит информацию о текущем CUDA-устройстве.
 *
 * @details Выводит название GPU, Compute Capability, объём глобальной
 *          памяти, количество мультипроцессоров, максимальное число
 *          потоков в блоке и характеристики поддержки Unified Memory.
 */
void printGPUInfo();

/**
 * @brief Проверяет поддержку Unified Memory prefetch текущим GPU.
 *
 * @details Проверка выполняется по свойству concurrentManagedAccess
 *          текущего CUDA-устройства.
 *
 * @return true, если устройство поддерживает concurrent managed access;
 *         false в противном случае или при ошибке получения свойств.
 */
bool isUnifiedPrefetchSupported();

/**
 * @brief Выполняет один эксперимент с выбранным алгоритмом и стратегией памяти.
 *
 * @param algorithm Вычислительный алгоритм умножения матриц.
 * @param strategy Стратегия размещения и передачи данных.
 * @param A Входная матрица A размером N x N.
 * @param B Входная матрица B размером N x N.
 * @param[out] C Вектор для записи результирующей матрицы C размером N x N.
 * @param N Размер стороны квадратных матриц.
 *
 * @return Структура Timing с измеренными временными характеристиками.
 *
 * @note Для потоковых стратегий матрица B передаётся на GPU один раз,
 *       а вычисление выполняется над частями матриц A и C.
 */
Timing runOneCase(
    Algorithm algorithm,
    MemoryStrategy strategy,
    const std::vector<BASE_TYPE>& A,
    const std::vector<BASE_TYPE>& B,
    std::vector<BASE_TYPE>& C,
    int N
);

/**
 * @brief Записывает заголовок CSV-файла с результатами экспериментов.
 * @param file Открытый поток выходного CSV-файла.
 */
void writeCSVHeader(std::ofstream& file);

/**
 * @brief Записывает результаты одного эксперимента в CSV-файл.
 * @param file Открытый поток выходного CSV-файла.
 * @param result Структура с результатами эксперимента.
 */
void writeCSVRow(
    std::ofstream& file,
    const BenchmarkResult& result
);

/**
 * @brief Вычисляет медиану набора измерений.
 *
 * @param values Набор значений для обработки.
 * @return Медиана отсортированного набора; 0.0, если набор пуст.
 *
 * @note Для чётного количества элементов возвращается среднее
 *       арифметическое двух центральных значений.
 */
double median(std::vector<double> values);

/**
 * @brief Возвращает минимальное значение набора измерений.
 *
 * @param values Набор значений для обработки.
 * @return Минимальное значение; 0.0, если набор пуст.
 */
double minimum(const std::vector<double>& values);

/** @brief Проверяет, что размер блока поддерживается (2, 4, 8, 16, 32). */
bool isSupportedBlockSize(int blockSize); 

/** @brief Задаёт размер блока для всех последующих запусков.  */
void setBlockSize(int blockSize); 

/** @brief Возвращает текущий размер блока. */
int getBlockSize();
