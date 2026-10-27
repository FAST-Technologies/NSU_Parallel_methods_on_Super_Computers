/**
 * @file main.cpp.
 * @brief Лабораторная работа №2: Параллельная реализация метода Якоби в 3D области.
 * @details Главная программа, решающая уравнение Пуассона методом Якоби с использованием
 *          гибридной модели параллелизма MPI + OpenMP.
 * @author F.A.S.T. DEVELOPMENT Refactored for educational purposes.
 * @version 1.0.
 */

#include "poisson_solver.hpp"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <chrono>

//#include <mpe.h> // Подключаем заголовок MPE

// Глобальные ID для фаз профилирования
int evt_begin_halo, evt_end_halo;
int evt_begin_comp_int, evt_end_comp_int;
int evt_begin_wait, evt_end_wait;
int evt_begin_comp_bnd, evt_end_comp_bnd;

double exact_solution(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double rhs_function(double x, double y, double z) {
    return 6.0 - A_PARAM * exact_solution(x, y, z);
}

int get_int_arg(int argc, char** argv, int arg_index, int default_value) {
    if (argc > arg_index) {
        try {
            return std::stoi(argv[arg_index]);
        } catch (const std::invalid_argument&) {
            std::cerr << "[WARNING] Аргумент '" << argv[arg_index] << "' не является числом. Используется значение по умолчанию: " << default_value << "\n";
        } catch (const std::out_of_range&) {
            std::cerr << "[WARNING] Аргумент '" << argv[arg_index] << "' выходит за пределы диапазона int. Используется значение по умолчанию: " << default_value << "\n";
        }
    }
    return default_value;
}

void initialize_grid(
    std::vector<double>& phi,
    std::vector<double>& phi_new,
    const DomainInfo& info,
    int Nx,
    int Ny,
    int Nz
)
{
    int stride_y = info.local_nz + 2;
    int stride_x = (info.local_ny + 2) * stride_y;

#pragma omp parallel for collapse(3)
    for (int i = 0; i < info.local_nx + 2; ++i) {
        for (int j = 0; j < info.local_ny + 2; ++j) {
            for (int k = 0; k < info.local_nz + 2; ++k) {
                int idx = i * stride_x + j * stride_y + k;

                // Глобальные индексы (учитывая сдвиг на 1 из-за ghost-слоя)
                int gi = info.coords[0] * info.local_nx + (i - 1);
                int gj = info.coords[1] * info.local_ny + (j - 1);
                int gk = k - 1;

                double x = X0 + gi * info.hx;
                double y = Y0 + gj * info.hy;
                double z = Z0 + gk * info.hz;

                // Граничные условия 1-го рода (Дирихле)
                if (gi == 0 || gi == Nx || gj == 0 || gj == Ny || gk == 0 || gk == Nz) {
                    double exact_val = exact_solution(x, y, z);
                    phi[idx] = exact_val;
                    phi_new[idx] = exact_val;
                } else {
                    phi[idx] = 0.0;
                    phi_new[idx] = 0.0;
                }
            }
        }
    }
}

void jacobi_step(
    std::vector<double>& phi,
    std::vector<double>& phi_new,
    const DomainInfo& info,
    int Nx,
    int Ny,
    int Nz,
    double& max_diff
)
{
    int stride_y = info.local_nz + 2;
    int stride_x = (info.local_ny + 2) * stride_y;

    // Буферы для обмена (размеры соответствуют граням без ghost-слоев)
    std::vector<double> send_left(info.local_ny * info.local_nz);
    std::vector<double> recv_left(info.local_ny * info.local_nz);
    std::vector<double> send_right(info.local_ny * info.local_nz);
    std::vector<double> recv_right(info.local_ny * info.local_nz);
    std::vector<double> send_bottom(info.local_nx * info.local_nz);
    std::vector<double> recv_bottom(info.local_nx * info.local_nz);
    std::vector<double> send_top(info.local_nx * info.local_nz);
    std::vector<double> recv_top(info.local_nx * info.local_nz);

    std::array<MPI_Request, 8> requests {};
    int req_count = 0;
    max_diff = 0.0;

    // ЭТАП 1: Упаковка и запуск обменов
//    MPE_Log_event(evt_begin_halo, 0, "start");

    // ------------------------------------------------------------------------
    // ЭТАП 1: Упаковка данных и запуск асинхронных обменов (Non-blocking)
    // ------------------------------------------------------------------------
#pragma omp parallel for collapse(2)
    for (int j = 0; j < info.local_ny; ++j) {
        for (int k = 0; k < info.local_nz; ++k) {
            send_left[j * info.local_nz + k] = phi[1 * stride_x + (j + 1) * stride_y + (k + 1)];
            send_right[j * info.local_nz + k] = phi[info.local_nx * stride_x + (j + 1) * stride_y + (k + 1)];
        }
    }
    MPI_Irecv(recv_left.data(), info.local_ny * info.local_nz, MPI_DOUBLE, info.left, 0, info.cart_comm, &requests[req_count++]);
    MPI_Irecv(recv_right.data(), info.local_ny * info.local_nz, MPI_DOUBLE, info.right, 1, info.cart_comm, &requests[req_count++]);
    MPI_Isend(send_left.data(), info.local_ny * info.local_nz, MPI_DOUBLE, info.left, 1, info.cart_comm, &requests[req_count++]);
    MPI_Isend(send_right.data(), info.local_ny * info.local_nz, MPI_DOUBLE, info.right, 0, info.cart_comm, &requests[req_count++]);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < info.local_nx; ++i) {
        for (int k = 0; k < info.local_nz; ++k) {
            send_bottom[i * info.local_nz + k] = phi[(i + 1) * stride_x + 1 * stride_y + (k + 1)];
            send_top[i * info.local_nz + k] = phi[(i + 1) * stride_x + info.local_ny * stride_y + (k + 1)];
        }
    }
    MPI_Irecv(recv_bottom.data(), info.local_nx * info.local_nz, MPI_DOUBLE, info.bottom, 2, info.cart_comm, &requests[req_count++]);
    MPI_Irecv(recv_top.data(), info.local_nx * info.local_nz, MPI_DOUBLE, info.top, 3, info.cart_comm, &requests[req_count++]);
    MPI_Isend(send_bottom.data(), info.local_nx * info.local_nz, MPI_DOUBLE, info.bottom, 3, info.cart_comm, &requests[req_count++]);
    MPI_Isend(send_top.data(), info.local_nx * info.local_nz, MPI_DOUBLE, info.top, 2, info.cart_comm, &requests[req_count++]);
//    MPE_Log_event(evt_end_halo, 0, "end");
    // ------------------------------------------------------------------------
    // ЭТАП 2: Вычисление ВНУТРЕННИХ точек (Перекрытие с коммуникациями!)
    // ------------------------------------------------------------------------
    // Эти точки не зависят от ghost-слоев, поэтому их можно считать, пока данные летят по сети.
    // ЭТАП 2: Вычисление ВНУТРЕННИХ точек
//    MPE_Log_event(evt_begin_comp_int, 0, "start");
#pragma omp parallel for collapse(3) reduction(max:max_diff)
    for (int i = 2; i < info.local_nx; ++i) {
        for (int j = 2; j < info.local_ny; ++j) {
            for (int k = 1; k < info.local_nz + 1; ++k) {
                int idx = i * stride_x + j * stride_y + k;
                int gi = info.coords[0] * info.local_nx + (i - 1);
                int gj = info.coords[1] * info.local_ny + (j - 1);
                int gk = k - 1;

                double x = X0 + gi * info.hx;
                double y = Y0 + gj * info.hy;
                double z = Z0 + gk * info.hz;
                double rho = rhs_function(x, y, z);

                double phi_x = (phi[(i - 1) * stride_x + j * stride_y + k] + phi[(i + 1) * stride_x + j * stride_y + k]) / (info.hx * info.hx);
                double phi_y = (phi[i * stride_x + (j - 1) * stride_y + k] + phi[i * stride_x + (j + 1) * stride_y + k]) / (info.hy * info.hy);
                double phi_z = (phi[i * stride_x + j * stride_y + (k - 1)] + phi[i * stride_x + j * stride_y + (k + 1)]) / (info.hz * info.hz);

                phi_new[idx] = (phi_x + phi_y + phi_z - rho) / info.denom;

                double diff = std::abs(phi_new[idx] - phi[idx]);
                if (diff > max_diff) max_diff = diff;
            }
        }
    }
//    MPE_Log_event(evt_end_comp_int, 0, "end");
    // ------------------------------------------------------------------------
    // ЭТАП 3: Ожидание завершения всех обменов
    // ------------------------------------------------------------------------
    // ЭТАП 3: Ожидание
//    MPE_Log_event(evt_begin_wait, 0, "start");
    MPI_Waitall(req_count, requests.data(), MPI_STATUSES_IGNORE);
//    MPE_Log_event(evt_end_wait, 0, "end");

    // ЭТАП 4 и 5: Распаковка и ГРАНИЧНЫЕ точки
//    MPE_Log_event(evt_begin_comp_bnd, 0, "start");
    // ------------------------------------------------------------------------
    // ЭТАП 4: Распаковка полученных данных в ghost-слои
    // ------------------------------------------------------------------------
#pragma omp parallel for collapse(2)
    for (int j = 0; j < info.local_ny; ++j) {
        for (int k = 0; k < info.local_nz; ++k) {
            phi[0 * stride_x + (j + 1) * stride_y + (k + 1)] = recv_left[j * info.local_nz + k];
            phi[(info.local_nx + 1) * stride_x + (j + 1) * stride_y + (k + 1)] = recv_right[j * info.local_nz + k];
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < info.local_nx; ++i) {
        for (int k = 0; k < info.local_nz; ++k) {
            phi[(i + 1) * stride_x + 0 * stride_y + (k + 1)] = recv_bottom[i * info.local_nz + k];
            phi[(i + 1) * stride_x + (info.local_ny + 1) * stride_y + (k + 1)] = recv_top[i * info.local_nz + k];
        }
    }

    // ------------------------------------------------------------------------
    // ЭТАП 5: Вычисление ГРАНИЧНЫХ точек локальной подобласти
    // ------------------------------------------------------------------------
    // Теперь, когда ghost-слои обновлены, можно посчитать точки на границе локального блока.
#pragma omp parallel for collapse(3) reduction(max:max_diff)
    for (int i = 1; i <= info.local_nx; ++i) {
        for (int j = 1; j <= info.local_ny; ++j) {
            for (int k = 1; k <= info.local_nz; ++k) {
                // Пропускаем внутренние точки, они уже посчитаны на Этапе 2
                if (i > 1 && i < info.local_nx && j > 1 && j < info.local_ny) continue;

                int idx = i * stride_x + j * stride_y + k;
                int gi = info.coords[0] * info.local_nx + (i - 1);
                int gj = info.coords[1] * info.local_ny + (j - 1);
                int gk = k - 1;

                // Глобальные границы не перевычисляются, они зафиксированы
                if (gi == 0 || gi == Nx || gj == 0 || gj == Ny || gk == 0 || gk == Nz) continue;

                double x = X0 + gi * info.hx;
                double y = Y0 + gj * info.hy;
                double z = Z0 + gk * info.hz;
                double rho = rhs_function(x, y, z);

                double phi_x = (phi[(i-1)*stride_x + j*stride_y + k] + phi[(i+1)*stride_x + j*stride_y + k]) / (info.hx * info.hx);
                double phi_y = (phi[i*stride_x + (j-1)*stride_y + k] + phi[i*stride_x + (j+1)*stride_y + k]) / (info.hy * info.hy);
                double phi_z = (phi[i*stride_x + j*stride_y + (k-1)] + phi[i*stride_x + j*stride_y + (k+1)]) / (info.hz * info.hz);

                phi_new[idx] = (phi_x + phi_y + phi_z - rho) / info.denom;

                double diff = std::abs(phi_new[idx] - phi[idx]);
                if (diff > max_diff) max_diff = diff;
            }
        }
    }
//    MPE_Log_event(evt_end_comp_bnd, 0, "end");
}

// ============================================================================
// Главная функция
// ============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размеры глобальной сетки
    int Nx = 128, Ny = 128, Nz = 128;
    if (argc > 3) {
        Nx = get_int_arg(argc, argv, 1, 128);
        Ny = get_int_arg(argc, argv, 2, 128);
        Nz = get_int_arg(argc, argv, 3, 128);
    }

    if (Nx <= 0 || Ny <= 0 || Nz <= 0) {
        if (rank == 0) {
            std::cerr << "Ошибка: размеры сетки должны быть положительными (Nx="
                      << Nx << ", Ny=" << Ny << ", Nz=" << Nz << ")" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Создание 2D декартовой топологии
    int dims[2] = {0, 0};
    int periods[2] = {0, 0}; // <-- ИСПРАВЛЕНО: 0 означает непериодические границы (Дирихле)
    MPI_Dims_create(size, 2, dims);

    if (dims[0] <= 0 || dims[1] <= 0) {
        if (rank == 0) {
            std::cerr << "Ошибка: MPI_Dims_create не смог разделить " << size
                      << " процессов на 2D сетку" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    DomainInfo info{};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &info.cart_comm);

	// Инициализация лога MPE
//	MPE_Init_log();

	// Получаем уникальные ID для наших фаз (вызывается на всех процессах синхронно)
//	evt_begin_halo = MPE_Log_get_event_number();
//	evt_end_halo = MPE_Log_get_event_number();
//	evt_begin_comp_int = MPE_Log_get_event_number();
//	evt_end_comp_int = MPE_Log_get_event_number();
//	evt_begin_wait = MPE_Log_get_event_number();
//	evt_end_wait = MPE_Log_get_event_number();
//	evt_begin_comp_bnd = MPE_Log_get_event_number();
//	evt_end_comp_bnd = MPE_Log_get_event_number();

	// Описываем состояния (название и цвет для Jumpshot)
//	MPE_Describe_state(evt_begin_halo, evt_end_halo, "1_Halo_Exchange", "red");
//	MPE_Describe_state(evt_begin_comp_int, evt_end_comp_int, "2_Compute_Internal", "green");
//	MPE_Describe_state(evt_begin_wait, evt_end_wait, "3_Waitall", "blue");
//	MPE_Describe_state(evt_begin_comp_bnd, evt_end_comp_bnd, "4_Compute_Boundary", "yellow");

    int cart_rank;
    MPI_Comm_rank(info.cart_comm, &cart_rank);
    MPI_Cart_coords(info.cart_comm, cart_rank, 2, info.coords);
    MPI_Cart_shift(info.cart_comm, 0, 1, &info.left, &info.right);
    MPI_Cart_shift(info.cart_comm, 1, 1, &info.bottom, &info.top);

    // Расчёт локальных размеров и шагов сетки
    info.local_nx = Nx / dims[0];
    info.local_ny = Ny / dims[1];
    info.local_nz = Nz;

    if (info.local_nx <= 0 || info.local_ny <= 0 || info.local_nz <= 0) {
        if (rank == 0) {
            std::cerr << "Ошибка: локальные размеры сетки слишком малы для "
                      << size << " процессов" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    info.hx = DX / Nx;
    info.hy = DY / Ny;
    info.hz = DZ / Nz;

    info.denom = 2.0 / (info.hx * info.hx) + 2.0 / (info.hy * info.hy) + 2.0 / (info.hz * info.hz) + A_PARAM;

    if (std::abs(info.denom) < 1e-15) {
        if (rank == 0) {
            std::cerr << "Ошибка: знаменатель формулы Якоби близок к нулю" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Выделение памяти с ghost-слоями: [local_nx + 2][local_ny + 2][local_nz + 2]
    int stride_y = info.local_nz + 2;
    int stride_x = (info.local_ny + 2) * stride_y;
    int total_size = (info.local_nx + 2) * stride_x;

    std::vector<double> phi(total_size, 0.0);
    std::vector<double> phi_new(total_size, 0.0);

    // Инициализация
    if (rank == 0) {
        std::cout << "Запуск расчета: сетка " << Nx << "x" << Ny << "x" << Nz
                  << " на " << size << " MPI процессах." << std::endl;
    }
    initialize_grid(phi, phi_new, info, Nx, Ny, Nz);

    // Итерационный процесс
    auto start_chrono_time = std::chrono::high_resolution_clock::now();
    double start_omp_time = omp_get_wtime();
    double start_mpi_time = MPI_Wtime();
    int iter = 0;
    double max_diff = 1.0;

    while (max_diff > EPSILON && iter < MAX_ITER) {
        jacobi_step(phi, phi_new, info, Nx, Ny, Nz, max_diff);

        // Глобальный сбор максимальной невязки
        double global_max_diff;
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        max_diff = global_max_diff;

        // Обновление массива phi = phi_new
#pragma omp parallel for
        for (int idx = 0; idx < total_size; ++idx) {
            phi[idx] = phi_new[idx];
        }

        iter++;
    }

    auto end_chrono_time = std::chrono::high_resolution_clock::now();
    double end_omp_time = omp_get_wtime();
    double end_mpi_time = MPI_Wtime();

    std::chrono::duration<double, std::milli> chrono_elapsed_time = end_chrono_time - start_chrono_time;
    double omp_elapsed_time = end_omp_time - start_omp_time;
    double elapsed_mpi_time = end_mpi_time - start_mpi_time;

    double tick = omp_get_wtick();

    // Вычисление точности решения (максимальное отклонение от точного решения)
    double max_error = 0.0;

#pragma omp parallel for collapse(3) reduction(max:max_error)
    for (int i = 1; i <= info.local_nx; ++i) {
        for (int j = 1; j <= info.local_ny; ++j) {
            for (int k = 1; k <= info.local_nz; ++k) {
                int idx = i * stride_x + j * stride_y + k;

                // Глобальные координаты
                int gi = info.coords[0] * info.local_nx + (i - 1);
                int gj = info.coords[1] * info.local_ny + (j - 1);
                int gk = k - 1;

                double x = X0 + gi * info.hx;
                double y = Y0 + gj * info.hy;
                double z = Z0 + gk * info.hz;

                // Точное решение
                double phi_exact = exact_solution(x, y, z);

                // Отклонение
                double error = std::abs(phi[idx] - phi_exact);
                if (error > max_error) max_error = error;
            }
        }
    }

    // Глобальный сбор максимальной ошибки
    double global_max_error;
    MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Конвергенция достигнута за " << iter << " итераций." << std::endl;
        std::cout << "Максимальная невязка: " << max_diff << std::endl;
        std::cout << "Максимальное отклонение от точного решения (Δ): " << global_max_error << "\n";
        std::cout << "Общее время счета (MPI TIME): " << elapsed_mpi_time << " sec." << std::endl;
        std::cout << "Общее время счета (Chrono TIME): " << chrono_elapsed_time.count() / 1000 << " sec." << std::endl;
        std::cout << "Общее время счета (OpenMP TIME) " << omp_elapsed_time << " sec." << std::endl;
        std::cout << "Точность таймера: " << tick << " секунд" << std::endl;;
    }

//    MPE_Finish_log("jacobi_profile.clog2");
    MPI_Comm_free(&info.cart_comm);
    MPI_Finalize();
    return 0;
}
