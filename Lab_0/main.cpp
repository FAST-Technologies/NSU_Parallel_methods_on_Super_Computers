/**
 * @file main.cpp
 * @brief Сборник примеров использования OpenMP для параллельного программирования.
 * @details Программа представляет собой интерактивное меню для запуска 19 различных
 *          примеров, демонстрирующих различные директивы и концепции OpenMP
 *          (редукция, задачи, векторизация, синхронизация и типичные ошибки).
 * @author Refactored for educational purposes
 * @version 2.0
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <queue>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <windows.h>
#include <omp.h>

#define NT 4

// ============================================================================
// Вспомогательные функции и функции для примеров
// ============================================================================

/**
 * @brief Рекурсивное суммирование массива с использованием задач OpenMP.
 * @param a Указатель на начало массива.
 * @param count Количество элементов для суммирования.
 * @return Сумма элементов.
 * @note Демонстрирует создание рекурсивных задач (tasks).
 */
double sum_7(double* a, int count) {
    if (count == 0) return 0;
    if (count == 1) return a[0];

    double res = 0;
    double s1 = 0, s2 = 0;
    int d = count/2;

#pragma omp task shared(s1)
    s1 = sum_7(a, d);
#pragma omp task shared(s2)
    s2 = sum_7(a + d, count - d);
#pragma omp taskwait
    res += s1 + s2;
    return res;
}

/**
 * @brief Векторизуемая функция сложения с унифицированным параметром.
 */
#pragma omp declare simd uniform(fact)
double add1(double a, double b, double fact) {
    return a + b + fact;
}

/**
 * @brief Векторизуемая функция с линейно изменяющимся индексом.
 */
#pragma omp declare simd uniform(a,b,fact) linear(i:1)
double add2(const double *a, const double *b, int i, double fact) {
    return a[i] + b[i] + fact;
}

/**
 * @brief Векторизуемая функция с линейно изменяющимися указателями.
 */
#pragma omp declare simd uniform(fact) linear(a,b:1)
double add3(const double *a, const double *b, double fact) {
    return *a + *b + fact;
}

/**
 * @brief Демонстрация векторизации цикла с вызовом SIMD-функций.
 * @param a Первый массив.
 * @param b Второй массив.
 * @param n Размер массивов.
 */
void work(double *a, double *b, int n) {
#pragma omp simd private(tmp)
    for (int i = 0; i < n; i++)
    {
        double tmp = add1(a[i], b[i], 1.0);
        a[i] = add2(a, b, i, 1.0) + tmp;
        a[i] = add3(&a[i], &b[i], 1.0);
    }
}

typedef std::vector<double> Vector;

double func(double x, double y) { return std::sin(x) * std::cos(y); }
double right(double x, double y) { return -2.0 * std::sin(x) * std::cos(y); }

/**
 * @brief Безопасный вывод строки в консоль из параллельной области.
 * @param msg Сообщение для вывода.
 */
void safe_print(const std::string& msg) {
#pragma omp critical
    {
        std::cout << msg << "\n";
    }
}

// ============================================================================
// Примеры (вынесены в отдельные функции)
// ============================================================================

/**
 * @brief Пример 1: Частные (private) и разделяемые (shared) переменные.
 */
void example_01() {
    std::cout << "\n--- Пример 1: Private и Shared переменные ---\n";
    char hello_string[] = "Hello World from thread";
    int id = 1234;
    int size = 0;
    std::cout << "Initial thread: id = " << id << ", size = " << size << std::endl;

#pragma omp parallel private(id) shared(hello_string)
    {
        size = omp_get_num_threads();
        id = omp_get_thread_num();
        safe_print(std::string(hello_string) + " " + std::to_string(id) + " (total threads: " + std::to_string(size) + ")");
    }
}

/**
 * @brief Пример 2: Использование редукции (reduction) для суммирования.
 */
void example_02() {
    std::cout << "\n--- Пример 2: Редукция (reduction) ---\n";
    int sum = 0;
    const int a_size = 100;
    std::vector<int> a(a_size);
    for (int i = 0; i < a_size; i++) a[i] = i;

#pragma omp parallel reduction(+:sum)
    {
        int id = omp_get_thread_num();
        int size = omp_get_num_threads();
        int integer_part = a_size / size;
        int remainder = a_size % size;
        int a_local_size = integer_part + ((id < remainder) ? 1 : 0);
        int start = integer_part * id + ((id < remainder) ? id : remainder);
        int end = start + a_local_size;

        int local_sum = 0;
        for (int i = start; i < end; i++) local_sum += a[i];
        sum += local_sum; // reduction сделает это безопасно

        safe_print("Thread " + std::to_string(id) + ", partial sum = " + std::to_string(local_sum));
    }
    std::cout << "Final sum = " << sum << std::endl;
}

/**
 * @brief Пример 3: Переопределение числа потоков через num_threads.
 * @note Исправлена ошибка оригинала: использовались неверные переменные id и size.
 */
void example_03() {
    std::cout << "\n--- Пример 3: Директива num_threads ---\n";
    omp_set_num_threads(4);

#pragma omp parallel num_threads(2)
    {
        int id_3 = omp_get_thread_num();
        if (0 == id_3) {
            int size_3 = omp_get_num_threads();
            std::cout << "Number of threads in this region = " << size_3 << std::endl;
        }
    }
}

/**
 * @brief Пример 4: Синхронизация потоков с помощью барьера (barrier).
 */
void example_04() {
    std::cout << "\n--- Пример 4: Барьер (barrier) ---\n";
    int num_threads = 8;
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int size_4 = omp_get_num_threads();
        int id_4 = omp_get_thread_num();
        for (int i = 0; i < size_4; i++) {
            if (i == id_4) {
                safe_print("Hello World from thread " + std::to_string(id_4) + " (total: " + std::to_string(size_4) + ")");
            }
#pragma omp barrier
        }
    }
}

/**
 * @brief Пример 5: Распараллеливание циклов (parallel for).
 * @note Исправлена ошибка оригинала: reduction применялся к sum, а не sum_5.
 */
void example_05() {
    std::cout << "\n--- Пример 5: Директива parallel for ---\n";
    int sum_5 = 0;
    const int a_size_5 = 100;
    std::vector<int> a_5(a_size_5);
    for (int i = 0; i < a_size_5; i++) a_5[i] = i;

    // Часть 1: Раздельные директивы
#pragma omp parallel reduction(+:sum_5)
    {
        int id_5 = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < a_size_5; i++) {
            sum_5 += a_5[i];
        }
        // Примечание: вывод partial sum здесь будет показывать промежуточное состояние reduction
    }
    std::cout << "#0 Final sum = " << sum_5 << std::endl;

    // Часть 2: Объединенная директива
    sum_5 = 0;
#pragma omp parallel for reduction(+:sum_5)
    for (int i = 0; i < a_size_5; i++) {
        sum_5 += a_5[i];
    }
    std::cout << "#1 Final sum = " << sum_5 << std::endl;
}

/**
 * @brief Пример 6: Недетерминированное поведение в секциях (sections).
 */
void example_06() {
    std::cout << "\n--- Пример 6: Секции (sections) ---\n";
    int section_count = 0;
    omp_set_num_threads(NT);

#pragma omp parallel
#pragma omp sections firstprivate(section_count)
    {
#pragma omp section
        {
            section_count++;
            safe_print("Section 1: section_count = " + std::to_string(section_count));
        }
#pragma omp section
        {
            section_count++;
            safe_print("Section 2: section_count = " + std::to_string(section_count));
        }
    }
}

/**
 * @brief Пример 7: Суммирование с помощью рекурсивных задач (tasks).
 */
void example_07() {
    std::cout << "\n--- Пример 7: Задачи (tasks) для суммирования ---\n";
    const int a_size_7 = 1000;
    double* a_7 = new double[a_size_7];

#pragma omp parallel for
    for (int i = 0; i < a_size_7; i++) a_7[i] = i;

    double res = 0;
#pragma omp parallel
#pragma omp single
    res = sum_7(a_7, a_size_7);

    std::cout << "Res = " << res << std::endl;
    delete[] a_7;
}

/**
 * @brief Пример 8: Демонстрация гонки данных (Data Race).
 * @warning Этот код содержит намеренную ошибку (гонку данных) для демонстрационных целей.
 */
void example_08() {
    std::cout << "\n--- Пример 8: Гонка данных (Data Race) ---\n";
    int var = 123;
#pragma omp parallel
    {
#pragma omp task
        {
            std::cout << "Task read var = " << var << std::endl;
        }
        var = 321; // Гонка: задача может прочитать старое или новое значение
    }
}

/**
 * @brief Пример 9: Демонстрация взаимной блокировки (Deadlock).
 * @warning Этот код содержит намеренную ошибку (deadlock) для демонстрационных целей.
 * @note Программа может зависнуть при выполнении этого примера.
 */
void example_09() {
    std::cout << "\n--- Пример 9: Взаимная блокировка (Deadlock) ---\n";
    std::cout << "ВНИМАНИЕ: Этот пример может зависнуть из-за deadlock. Нажмите Ctrl+C для прерывания.\n";

#pragma omp parallel
    {
#pragma omp task
        {
#pragma omp critical
            {
                std::cerr << "Задача 1 вошла в критическую секцию" << std::endl;
            }
        }
#pragma omp critical
        {
            std::cerr << "Главный поток вошел в критическую секцию" << std::endl;
#pragma omp task
            std::cerr << "Задача 2 создана внутри критической секции" << std::endl;
        }
    }
}

/**
 * @brief Пример 10: Векторизация вложенных циклов (simd collapse).
 */
void example_10() {
    std::cout << "\n--- Пример 10: Векторизация (simd collapse) ---\n";
    float a_10[80], b_10[80], sum_10 = 0;
    for (int i = 0; i < 80; i++) a_10[i] = b_10[i] = i;

#pragma omp simd collapse(2) reduction(+:sum_10)
    for (int i = 0; i < 40; i++) {
        for (int j = 0; j < 2; j++) {
            sum_10 += a_10[i * 2 + j] + b_10[i * 2 + j];
        }
    }
    std::cout << "Sum = " << sum_10 << std::endl;
}

/**
 * @brief Пример 11: Использование функций в векторизуемых циклах.
 * @note Исправлена ошибка оригинала: обращение к a вместо a_11.
 */
void example_11() {
    std::cout << "\n--- Пример 11: Векторизация с вызовом функций ---\n";
    const int N = 32;
    std::vector<double> a_11(N), b_11(N);
    for (int i = 0; i < N; i++) {
        a_11[i] = i;
        b_11[i] = N - i;
    }
    work(a_11.data(), b_11.data(), N);

    std::cout << "First 5 results: ";
    for (int i = 0; i < 5; i++) {
        std::cout << a_11[i] << " ";
    }
    std::cout << "...\n";
}

/**
 * @brief Пример 12: Гонка данных при суммировании без редукции.
 * @warning Намеренная ошибка для демонстрации некорректного суммирования.
 */
void example_12() {
    std::cout << "\n--- Пример 12: Гонка данных при суммировании ---\n";
    const int size_12 = 1000;
    std::vector<int> a_12(size_12);

#pragma omp parallel for
    for (int i = 0; i < size_12; i++) a_12[i] = i;

    int sum_12 = 0;
#pragma omp parallel for
    for(int i = 0; i < size_12; i++) {
        sum_12 += a_12[i]; // ОШИБКА: гонка данных!
    }
    std::cout << "Incorrect Sum (due to race condition) = " << sum_12 << " (Expected: " << (size_12 * (size_12 - 1)) / 2 << ")\n";
}

/**
 * @brief Пример 13: Deadlock при использовании явных блокировок (locks).
 * @warning Намеренная ошибка. Программа может зависнуть.
 */
void example_13() {
    std::cout << "\n--- Пример 13: Deadlock с явными блокировками ---\n";
    std::cout << "ВНИМАНИЕ: Этот пример может зависнуть.\n";

    const int size_13 = 10;
    omp_lock_t queue_lock, io_lock;
    omp_init_lock(&queue_lock);
    omp_init_lock(&io_lock);
    std::queue<int> queue;

#pragma omp parallel sections
    {
#pragma omp section
        {
            for (int i = 0; i < size_13; i++) {
                omp_set_lock(&queue_lock);
                omp_set_lock(&io_lock); // Потенциальный deadlock, если порядок захвата не совпадает
                queue.push(i);
                std::cout << "Put " << i << " into queue" << std::endl;
                omp_unset_lock(&io_lock);
                omp_unset_lock(&queue_lock);
            }
        }
#pragma omp section
        {
            bool working = true;
            while (working) {
                omp_set_lock(&io_lock);
                omp_set_lock(&queue_lock); // Обратный порядок захвата блокировок -> DEADLOCK
                if (!queue.empty()) {
                    int value = queue.front();
                    queue.pop();
                    std::cout << "Get " << value << " from queue" << std::endl;
                    working = (value < size_13 - 1);
                } else {
                    working = false;
                }
                omp_unset_lock(&queue_lock);
                omp_unset_lock(&io_lock);
            }
        }
    }
    omp_destroy_lock(&queue_lock);
    omp_destroy_lock(&io_lock);
}

/**
 * @brief Пример 14: Исправление Примера 12 с помощью редукции.
 */
void example_14() {
    std::cout << "\n--- Пример 14: Исправление гонки данных (reduction) ---\n";
    const int size_14 = 1000;
    std::vector<int> a_14(size_14);

#pragma omp parallel for
    for (int i = 0; i < size_14; i++) a_14[i] = i;

    int sum_14 = 0;
#pragma omp parallel for reduction(+:sum_14)
    for (int i = 0; i < size_14; i++) {
        sum_14 += a_14[i];
    }
    std::cout << "Correct Sum = " << sum_14 << std::endl;
}

/**
 * @brief Пример 15: Решение уравнения Пуассона методом Якоби (базовый).
 * @note Размер сетки уменьшен до 200 для быстрого выполнения в рамках меню.
 */
void example_15() {
    std::cout << "\n--- Пример 15: Уравнение Пуассона (Якоби, базовый) ---\n";
    const int N_15 = 200;
    const double eps = 1e-6;
    const int max_iter = 1000;
    const double h = 1.0 / N_15;

    Vector A(N_15 * N_15, 0.0), A_next(N_15 * N_15, 0.0), f(N_15 * N_15, 0.0);

    for(int j = 0; j < N_15; j++) {
        for(int i = 0; i < N_15; i++) {
            if(i == 0 || j == 0 || i == N_15-1 || j == N_15-1) {
                A[i*N_15 + j] = func(i*h, j*h);
            }
            f[i*N_15 + j] = right(i*h, j*h);
        }
    }

    double norm = 0.0;
    int iter = 0;
    do {
        norm = 0.0;
#pragma omp parallel for reduction(max:norm)
        for (int j = 1; j < N_15 - 1; j++) {
            for (int i = 1; i < N_15 - 1; i++) {
                A_next[i*N_15 + j] = (A[(i - 1)*N_15 + j] + A[(i + 1)*N_15 + j] +
                                      A[i*N_15 + j - 1] + A[i*N_15 + j + 1] - h*h*f[i*N_15 + j]) / 4.0;
                norm = std::fmax(norm, std::fabs(A_next[i*N_15 + j] - A[i*N_15 + j]));
            }
        }
        A = A_next;
        iter++;
    } while ((norm > eps) && (iter < max_iter));

    std::cout << "Norm = " << norm << ", iter = " << iter << std::endl;
}

/**
 * @brief Пример 16: Уравнение Пуассона с перестановкой циклов (i и j).
 */
void example_16() {
    std::cout << "\n--- Пример 16: Уравнение Пуассона (перестановка циклов) ---\n";
    const int N_16 = 200;
    const double eps_16 = 1e-6;
    const int max_iter_16 = 1000;
    const double h_16 = 1.0 / N_16;

    Vector A_16(N_16 * N_16, 0.0), A_next_16(N_16 * N_16, 0.0), f_16(N_16 * N_16, 0.0);

    for(int i = 0; i < N_16; i++) {
        for(int j = 0; j < N_16; j++) {
            if(i == 0 || j == 0 || i == N_16-1 || j == N_16-1) A_16[i*N_16 + j] = func(i*h_16, j*h_16);
            f_16[i*N_16 + j] = right(i*h_16, j*h_16);
        }
    }

    double norm_16 = 0.0;
    int iter_16 = 0;
    do {
        norm_16 = 0.0;
#pragma omp parallel for reduction(max:norm_16)
        for (int i = 1; i < N_16 - 1; i++) {
            for (int j = 1; j < N_16 - 1; j++) {
                A_next_16[i*N_16 + j] = (A_16[(i - 1)*N_16 + j] + A_16[(i + 1)*N_16 + j] +
                                         A_16[i*N_16 + j - 1] + A_16[i*N_16 + j + 1] - h_16*h_16*f_16[i*N_16 + j]) / 4.0;
                norm_16 = std::fmax(norm_16, std::fabs(A_next_16[i*N_16 + j] - A_16[i*N_16 + j]));
            }
        }
        A_16 = A_next_16;
        iter_16++;
    } while ((norm_16 > eps_16) && (iter_16 < max_iter_16));

    std::cout << "Norm = " << norm_16 << ", iter = " << iter_16 << std::endl;
}

/**
 * @brief Пример 17: Уравнение Пуассона с распараллеливанием копирования массива.
 */
void example_17() {
    std::cout << "\n--- Пример 17: Уравнение Пуассона (параллельное копирование) ---\n";
    const int N_17 = 200;
    const double eps_17 = 1e-6;
    const int max_iter_17 = 1000;
    const double h_17 = 1.0 / N_17;

    Vector A_17(N_17 * N_17, 0.0), A_next_17(N_17 * N_17, 0.0), f_17(N_17 * N_17, 0.0);

#pragma omp parallel for collapse(2)
    for(int i = 0; i < N_17; i++) {
        for(int j = 0; j < N_17; j++) {
            if(i == 0 || j == 0 || i == N_17-1 || j == N_17-1) A_17[i*N_17 + j] = func(i*h_17, j*h_17);
            f_17[i*N_17 + j] = right(i*h_17, j*h_17);
        }
    }

    double norm_17 = 0.0;
    int iter_17 = 0;
    do {
        norm_17 = 0.0;
#pragma omp parallel for reduction(max:norm_17)
        for (int i = 1; i < N_17 - 1; i++) {
            for (int j = 1; j < N_17 - 1; j++) {
                A_next_17[i*N_17 + j] = (A_17[(i - 1)*N_17 + j] + A_17[(i + 1)*N_17 + j] +
                                         A_17[i*N_17 + j - 1] + A_17[i*N_17 + j + 1] - h_17*h_17*f_17[i*N_17 + j]) / 4.0;
                norm_17 = std::fmax(norm_17, std::fabs(A_next_17[i*N_17 + j] - A_17[i*N_17 + j]));
            }
        }
#pragma omp parallel for collapse(2)
        for(int i = 0; i < N_17; i++) {
            for(int j = 0; j < N_17; j++) {
                A_17[i*N_17 + j] = A_next_17[i*N_17 + j];
            }
        }
        iter_17++;
    } while ((norm_17 > eps_17) && (iter_17 < max_iter_17));

    std::cout << "Norm = " << norm_17 << ", iter = " << iter_17 << std::endl;
}

/**
 * @brief Пример 18: Уравнение Пуассона с динамическим планированием (schedule dynamic).
 */
void example_18() {
    std::cout << "\n--- Пример 18: Уравнение Пуассона (dynamic schedule) ---\n";
    const int N_18 = 200;
    const double eps_18 = 1e-6;
    const int max_iter_18 = 1000;
    const double h_18 = 1.0 / N_18;

    Vector A_18(N_18 * N_18, 0.0), A_next_18(N_18 * N_18, 0.0), f_18(N_18 * N_18, 0.0);

#pragma omp parallel for collapse(2)
    for(int i = 0; i < N_18; i++) {
        for(int j = 0; j < N_18; j++) {
            if(i == 0 || j == 0 || i == N_18-1 || j == N_18-1) A_18[i*N_18 + j] = func(i*h_18, j*h_18);
            f_18[i*N_18 + j] = right(i*h_18, j*h_18);
        }
    }

    double norm_18 = 0.0;
    int iter_18 = 0;
    do {
        norm_18 = 0.0;
#pragma omp parallel for reduction(max:norm_18) schedule(dynamic, 10)
        for (int i = 1; i < N_18 - 1; i++) {
            for (int j = 1; j < N_18 - 1; j++) {
                A_next_18[i*N_18 + j] = (A_18[(i - 1)*N_18 + j] + A_18[(i + 1)*N_18 + j] +
                                         A_18[i*N_18 + j - 1] + A_18[i*N_18 + j + 1] - h_18*h_18*f_18[i*N_18 + j]) / 4.0;
                norm_18 = std::fmax(norm_18, std::fabs(A_next_18[i*N_18 + j] - A_18[i*N_18 + j]));
            }
        }
#pragma omp parallel for collapse(2) schedule(dynamic, 10)
        for(int i = 0; i < N_18; i++) {
            for(int j = 0; j < N_18; j++) {
                A_18[i*N_18 + j] = A_next_18[i*N_18 + j];
            }
        }
        iter_18++;
    } while ((norm_18 > eps_18) && (iter_18 < max_iter_18));

    std::cout << "Norm = " << norm_18 << ", iter = " << iter_18 << std::endl;
}

/**
 * @brief Пример 19: Зависимости между параллельными циклами.
 */
void example_19() {
    std::cout << "\n--- Пример 19: Параллельные циклы с зависимостями ---\n";
    std::vector<int> a_19(100), b_19(100);

    for(int i = 0; i < 100; i++) b_19[i] = i;

#pragma omp parallel for
    for(int i = 0; i < 100; i++) {
        a_19[i] = b_19[i];
        b_19[i] = 2 * a_19[i];
    }

    int result_19 = 0;
#pragma omp parallel for reduction(+:result_19)
    for(int i = 0; i < 100; i++) {
        result_19 += (a_19[i] + b_19[i]);
    }
    std::cout << "Result = " << result_19 << std::endl;
}

// ============================================================================
// Главное меню и точка входа
// ============================================================================

/**
 * @brief Отображение главного меню программы.
 */
void print_menu() {
    std::cout << "\n=======================================================\n";
    std::cout << "          МЕНЮ ПРИМЕРОВ OPENMP\n";
    std::cout << "=======================================================\n";
    std::cout << " 1. Private и Shared переменные\n";
    std::cout << " 2. Редукция (reduction)\n";
    std::cout << " 3. Директива num_threads\n";
    std::cout << " 4. Барьер (barrier)\n";
    std::cout << " 5. Директива parallel for\n";
    std::cout << " 6. Секции (sections)\n";
    std::cout << " 7. Задачи (tasks) для суммирования\n";
    std::cout << " 8. [ОШИБКА] Гонка данных (Data Race)\n";
    std::cout << " 9. [ОШИБКА] Взаимная блокировка (Deadlock)\n";
    std::cout << "10. Векторизация (simd collapse)\n";
    std::cout << "11. Векторизация с вызовом функций\n";
    std::cout << "12. [ОШИБКА] Гонка данных при суммировании\n";
    std::cout << "13. [ОШИБКА] Deadlock с явными блокировками\n";
    std::cout << "14. Исправление гонки данных (reduction)\n";
    std::cout << "15. Уравнение Пуассона (базовый)\n";
    std::cout << "16. Уравнение Пуассона (перестановка циклов)\n";
    std::cout << "17. Уравнение Пуассона (параллельное копирование)\n";
    std::cout << "18. Уравнение Пуассона (dynamic schedule)\n";
    std::cout << "19. Параллельные циклы с зависимостями\n";
    std::cout << " 0. Выход из программы\n";
    std::cout << "=======================================================\n";
    std::cout << "Выберите пункт меню (0-19): ";
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(CP_UTF8);

    // Устанавливаем разумное количество потоков по умолчанию для демонстраций
    omp_set_num_threads(4);

    int choice = -1;
    bool exit_program = false;

    while (!exit_program) {
        print_menu();

        // Проверка ввода: защита от нечисловых символов
        if (!(std::cin >> choice)) {
            std::cin.clear(); // Очистка флага ошибки
            std::cin.ignore(32767, '\n'); // Очистка буфера ввода
            std::cout << "\n[ОШИБКА] Неверный ввод. Пожалуйста, введите число от 0 до 19.\n";
            continue;
        }

        switch (choice) {
            case 0:
                exit_program = true;
                std::cout << "Завершение работы. До свидания!\n";
                break;
            case 1: example_01(); break;
            case 2: example_02(); break;
            case 3: example_03(); break;
            case 4: example_04(); break;
            case 5: example_05(); break;
            case 6: example_06(); break;
            case 7: example_07(); break;
            case 8: example_08(); break;
            case 9: example_09(); break;
            case 10: example_10(); break;
            case 11: example_11(); break;
            case 12: example_12(); break;
            case 13: example_13(); break;
            case 14: example_14(); break;
            case 15: example_15(); break;
            case 16: example_16(); break;
            case 17: example_17(); break;
            case 18: example_18(); break;
            case 19: example_19(); break;
            default:
                std::cout << "\n[ОШИБКА] Пункт '" << choice << "' не существует. Попробуйте снова.\n";
                break;
        }

        if (!exit_program) {
            std::cout << "\nНажмите Enter для продолжения...";
            std::cin.ignore(32767, '\n');
            std::cin.get();
        }
    }
    return 0;
}
