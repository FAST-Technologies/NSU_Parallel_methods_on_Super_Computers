#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <windows.h>
#include <omp.h>

void safe_print(const std::string& msg) {
#pragma omp critical
    {
        std::cout << msg << "\n";
    }
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    omp_set_num_threads(20);

    int max_threads = omp_get_max_threads();
    std::cout << "=== Среда OpenMP ===\n";
    std::cout << "Доступно максимум потоков: " << max_threads << "\n\n";

    int num_threads = omp_get_num_threads();
    std::cout << "Число доступных потоков: " << num_threads << "\n\n";

#pragma omp parallel
    {
        int current_threads = omp_get_num_threads();
#pragma omp single
        {
            std::cout << "Фактическое число потоков в этой области: " << current_threads << "\n";
        }
    }

    // ==========================================
    // 0. Первоначальные примеры
    // ==========================================
#pragma omp parallel
    {
#pragma omp for schedule(static, 10)
        for (int i = 0; i < 20; i++) {
            std::cout << "Iteration number: " << i << " Thread number: " << omp_get_thread_num() << std::endl;
        }
    };

    // ==========================================
    // 1. Синхронизация (Critical Section)
    // ==========================================
    std::cout << "--- 1. Критические секции (Исправляем гонку данных) ---\n";
    int id;
#pragma omp parallel private(id)
    {
        id = omp_get_thread_num();
        // Без critical секции строки бы перемешались
        safe_print("Привет из потока " + std::to_string(id));
    }

    // ==========================================
    // 2. Редукция (Reduction) и Замер времени
    // ==========================================
    std::cout << "\n--- 2. Редукция (Сумма массива) ---\n";
    const size_t N = 50000000;
    std::vector<long long> data(N, 1); // Массив из 50 млн единиц

    long long sum = 0;
    double start_time = omp_get_wtime(); // Засекаем время

    // reduction(+:sum) автоматически создает локальные копии sum для каждого потока,
    // а в конце корректно их складывает.
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
        sum += data[i];
    }

    double end_time = omp_get_wtime();
    std::cout << "Сумма: " << sum << "\n";
    std::cout << "Время выполнения: " << (end_time - start_time) * 1000.0 << " мс\n";

    // ==========================================
    // 3. Балансировка нагрузки (Schedule)
    // ==========================================
    std::cout << "\n--- 3. Динамическое планирование (Dynamic Schedule) ---\n";
    // Представим, что некоторые итерации выполняются дольше других
#pragma omp parallel for schedule(dynamic, 5)
    for (int i = 0; i < 20; ++i) {
        int thread_id = omp_get_thread_num();
        // Имитация разной нагрузки
        int sleep_time = (i % 3 == 0) ? 100 : 10;

        // В реальном коде тут были бы вычисления.
        // Для примера просто сделаем задержку (в микросекундах)
        // omp_set_nested(1); // Для старых версий, сейчас не обязательно

        safe_print("Поток " + std::to_string(thread_id) + " делает итерацию " +
                   std::to_string(i) + " (нагрузка: " + std::to_string(sleep_time) + ")");
    }

    // ==========================================
    // 4. Вложенные циклы (Collapse)
    // ==========================================
    std::cout << "\n--- 4. Сворачивание вложенных циклов (Collapse) ---\n";
    const int rows = 4, cols = 4;
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols, 0));

    // collapse(2) объединяет два цикла в один большой, чтобы потоки брали
    // и строки, и столбцы, а не только внешние строки.
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = omp_get_thread_num();
        }
    }

    std::cout << "Матрица (значения = ID потоков):\n";
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }

    // ==========================================
    // 5. Задачи (Tasks) - Неструктурированный параллелизм
    // ==========================================
    std::cout << "\n--- 5. Задачи (Tasks) ---\n";
#pragma omp parallel
    {
        // Только ОДИН поток создаст задачи, но выполнять их будут ВСЕ потоки
#pragma omp single
        {
#pragma omp task
            {
                safe_print("Задача 1: Вычисление синуса (поток " + std::to_string(omp_get_thread_num()) + ")");
                std::sin(1.5);
            }

#pragma omp task
            {
                safe_print("Задача 2: Вычисление косинуса (поток " + std::to_string(omp_get_thread_num()) + ")");
                std::cos(1.5);
            }

            // Ждем завершения всех задач внутри single
#pragma omp taskwait
            safe_print("Все задачи завершены!");
        }
    }
    return 0;
}
