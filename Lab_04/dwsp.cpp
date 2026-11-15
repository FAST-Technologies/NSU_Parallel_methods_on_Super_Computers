/**
 * @file dwsp.cpp
 * @brief Основная реализация протокола динамического распределения задач DWSP.
 * @details
 * Проект DWSP (Dynamic Work Stealing Protocol) реализует гибридную
 * параллельную обработку задач с использованием MPI и POSIX Threads.
 *
 * Каждый MPI-процесс содержит:
 * - несколько Worker-потоков для выполнения задач;
 * - один Fetcher-поток для получения задач от других процессов;
 * - один Responder-поток для обработки входящих запросов.
 *
 * Для синхронизации потоков используются мьютекс и две условные переменные.
 *
 * @author F.A.S.T. DEVELOPMENT aka Ямщиков Владимир, группа 26162.
 * @date 15.11.2026.
 * @version 2.0
 */

#include "dwsp.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdlib>

// ============================================================
// Глобальные переменные
// ============================================================
std::queue<Job> jobQueue;
MPI_Comm replyComm;
pthread_mutex_t mutex;
pthread_cond_t c_jobAdded;
pthread_cond_t c_needToFetch;
bool shutdownRequested = false;

int rank = -1;
int size = 0;
int totalJobs = 0;

std::atomic<int> jobsCompleted{0};
std::atomic<int> jobsSent{0};
std::atomic<int> jobsReceived{0};

// ============================================================
// Реализация структуры Job
// ============================================================
Job::Job() : id(-1), durationMs(0) {}

Job::Job(int jobId, int duration)
        : id(jobId), durationMs(duration) {}

void Job::process() const {
    std::this_thread::sleep_for(
            std::chrono::milliseconds(durationMs)
    );
}

// ============================================================
// Инициализация очереди
// ============================================================
void initializeJobs(int baseMs) {
    for (int id = 0; id < totalJobs; ++id) {
        if (id % size == rank) {
            int duration =
                    baseMs + (id * 37) % (baseMs * 3 + 1);
            jobQueue.push(Job(id, duration));
        }
    }
}

// ============================================================
// Завершение процесса
// ============================================================
void requestShutdown() {
    pthread_mutex_lock(&mutex);
    shutdownRequested = true;
    pthread_cond_broadcast(&c_jobAdded);
    pthread_cond_broadcast(&c_needToFetch);
    pthread_mutex_unlock(&mutex);
}

// ============================================================
// Worker — выполнение задач из локальной очереди
// ============================================================
void* worker(void*) {
    while (true) {
        Job job;
        pthread_mutex_lock(&mutex);
        // Ожидание новой работы или завершения.
        while (jobQueue.empty() && !shutdownRequested) {
            pthread_cond_wait(&c_jobAdded, &mutex);
        }
        // Завершение потока.
        if (jobQueue.empty() && shutdownRequested) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        // Получаем задачу.
        job = jobQueue.front();
        jobQueue.pop();
        // Если очередь опустела,
        // пробуждаем Fetcher.
        if (jobQueue.empty()) {
            pthread_cond_signal(&c_needToFetch);
        }
        pthread_mutex_unlock(&mutex);
        job.process();
        jobsCompleted.fetch_add(1);
        int completedId = job.id;
        MPI_Send(
                &completedId,
                1,
                MPI_INT,
                0,
                TAG_JOB_DONE,
                MPI_COMM_WORLD
        );
        std::cout
                << "[Rank " << rank << "] "
                << "Completed job " << job.id
                << std::endl;
    }
    return nullptr;
}

// ============================================================
// Fetcher — получение задач от других MPI-процессов
// ============================================================
void* fetcher(void*) {
    while (true) {
        pthread_mutex_lock(&mutex);
        while (!jobQueue.empty() && !shutdownRequested) {
            pthread_cond_wait(&c_needToFetch, &mutex);
        }
        if (shutdownRequested) {
            pthread_mutex_unlock(&mutex);
            return nullptr;
        }
        pthread_mutex_unlock(&mutex);
        bool receivedAnyWork = false;
        for (int target = 0; target < size; ++target) {
            if (target == rank) {
                continue;
            }
            pthread_mutex_lock(&mutex);
            int currentSize =
                    static_cast<int>(jobQueue.size());
            bool shouldStop = shutdownRequested;
            pthread_mutex_unlock(&mutex);
            if (shouldStop) {
                return nullptr;
            }
            int requestedJobs =
                    HIGH_WATERMARK - currentSize;
            if (requestedJobs <= 0) {
                break;
            }
            MPI_Send(
                    &requestedJobs,
                    1,
                    MPI_INT,
                    target,
                    TAG_REQUEST,
                    MPI_COMM_WORLD
            );
            int receivedCount = 0;
            MPI_Recv(
                    &receivedCount,
                    1,
                    MPI_INT,
                    target,
                    TAG_REPLY_COUNT,
                    replyComm,
                    MPI_STATUS_IGNORE
            );
            if (receivedCount <= 0) {
                continue;
            }
            // Получаем массив:
            // [job_id, duration, job_id, duration, ...]
            std::vector<int> data(receivedCount * 2);
            MPI_Recv(
                    data.data(),
                    receivedCount * 2,
                    MPI_INT,
                    target,
                    TAG_REPLY_DATA,
                    replyComm,
                    MPI_STATUS_IGNORE
            );
            // Добавляем полученные задачи в очередь.
            pthread_mutex_lock(&mutex);
            for (int i = 0; i < receivedCount; ++i) {
                Job job(
                        data[2 * i],
                        data[2 * i + 1]
                );
                jobQueue.push(job);
                std::cout
                        << "[Rank " << rank << "] "
                        << "Received job " << job.id
                        << " from Rank " << target
                        << std::endl;
            }
            jobsReceived.fetch_add(receivedCount);
            receivedAnyWork = true;
            // Пробуждаем Worker.
            pthread_cond_broadcast(&c_jobAdded);
            pthread_mutex_unlock(&mutex);
        }

        if (!receivedAnyWork) {
            std::cout
                    << "[Rank " << rank << "] "
                    << "Fetcher is IDLE"
                    << std::endl;
            int idleMessage = 1;
            MPI_Send(
                    &idleMessage,
                    1,
                    MPI_INT,
                    0,
                    TAG_FETCHER_IDLE,
                    MPI_COMM_WORLD
            );
            return nullptr;
        }
    }
    return nullptr;
}

// ============================================================
// Responder — обработка входящих MPI-сообщений
// ============================================================
void* responder(void*) {
    int completedCount = 0;
    int idleCount = 0;
    std::vector<bool> completedJobs(totalJobs, false);
    while (true) {
        int message = 0;
        MPI_Status status;
        MPI_Recv(
                &message,
                1,
                MPI_INT,
                MPI_ANY_SOURCE,
                MPI_ANY_TAG,
                MPI_COMM_WORLD,
                &status
        );
        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        // Запрос на получение задач
        if (tag == TAG_REQUEST) {
            int requestedJobs = message;
            std::vector<int> data;
            int transferCount = 0;
            pthread_mutex_lock(&mutex);
            int available =
                    static_cast<int>(jobQueue.size());
            if (available > 1) {
                transferCount = std::min(
                        requestedJobs,
                        available / 2
                );
            }

            for (int i = 0; i < transferCount; ++i) {
                Job job = jobQueue.front();
                jobQueue.pop();
                data.push_back(job.id);
                data.push_back(job.durationMs);
            }
            jobsSent.fetch_add(transferCount);
            if (jobQueue.empty()) {
                pthread_cond_signal(&c_needToFetch);
            }
            pthread_mutex_unlock(&mutex);
            MPI_Send(
                    &transferCount,
                    1,
                    MPI_INT,
                    source,
                    TAG_REPLY_COUNT,
                    replyComm
            );
            // Отправляем сами задачи.
            if (transferCount > 0) {
                MPI_Send(
                        data.data(),
                        transferCount * 2,
                        MPI_INT,
                        source,
                        TAG_REPLY_DATA,
                        replyComm
                );
            }
            std::cout
                    << "[Rank " << rank << "] "
                    << "Sent " << transferCount
                    << " jobs to Rank " << source
                    << std::endl;
        }

        // Сообщение о завершении задачи
        else if (tag == TAG_JOB_DONE) {
            if (rank == 0) {
                if (message < 0 || message >= totalJobs) {
                    std::cerr
                            << "Invalid job ID: " << message
                            << std::endl;
                }
                else if (completedJobs[message]) {
                    std::cerr
                            << "Duplicate completion of job "
                            << message
                            << std::endl;
                }
                else {
                    completedJobs[message] = true;
                    ++completedCount;
                }

                if (completedCount % 10 == 0 ||
                    completedCount == totalJobs) {
                    std::cout
                            << "[Coordinator] Completed "
                            << completedCount
                            << " / " << totalJobs
                            << std::endl;
                }
            }
        }
        else if (tag == TAG_FETCHER_IDLE) {
            if (rank == 0) {
                ++idleCount;
                std::cout
                        << "[Coordinator] IDLE received: "
                        << idleCount << " / " << size
                        << std::endl;
            }
        }

        // Получение команды завершения
        else if (tag == TAG_TERMINATE) {
            requestShutdown();
            break;
        }

        // Координатор Rank 0
        if (rank == 0) {
            if (completedCount >= totalJobs &&
                idleCount >= size) {
                std::cout
                        << "\n[Coordinator] "
                        << "All jobs completed!"
                        << std::endl;
                int terminateMessage = 1;
                for (int r = 1; r < size; ++r) {
                    MPI_Send(
                            &terminateMessage,
                            1,
                            MPI_INT,
                            r,
                            TAG_TERMINATE,
                            MPI_COMM_WORLD
                    );
                }
                // Завершаем локальные потоки.
                requestShutdown();
                break;
            }
        }
    }
    return nullptr;
}

/**
 * @brief Точка входа в программу DWSP.
 *
 * @details
 * Выполняет следующие этапы:
 * 1. Инициализация MPI с поддержкой MPI_THREAD_MULTIPLE.
 * 2. Определение номера и количества MPI-процессов.
 * 3. Создание коммуникатора для ответов.
 * 4. Инициализация очереди задач и средств синхронизации.
 * 5. Создание Worker, Fetcher и Responder.
 * 6. Ожидание завершения потоков.
 * 7. Сбор статистики с помощью MPI_Reduce.
 * 8. Освобождение ресурсов и завершение MPI.
 *
 * @param argc Количество аргументов командной строки.
 * @param argv Аргументы командной строки.
 * @return 0 при успешном завершении.
 */
int main(int argc, char** argv) {
    int required = MPI_THREAD_MULTIPLE;
    int provided = MPI_THREAD_SINGLE;

    MPI_Init_thread(
            &argc,
            &argv,
            required,
            &provided
    );

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            std::cerr
                    << "MPI_THREAD_MULTIPLE is not supported!"
                    << std::endl;
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Comm_dup(MPI_COMM_WORLD, &replyComm);

    totalJobs = 40;
    int baseMs = 100;

    if (argc >= 2) {
        totalJobs = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        baseMs = std::atoi(argv[2]);
    }

    if (totalJobs < 0 || baseMs <= 0) {
        if (rank == 0) {
            std::cerr
                    << "Invalid program arguments."
                    << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    // Synchronization
    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(
            &c_jobAdded,
            nullptr
    );
    pthread_cond_init(
            &c_needToFetch,
            nullptr
    );

    // Initial jobs
    initializeJobs(baseMs);
    std::cout
            << "[Rank " << rank << "] "
            << "Initial jobs: "
            << jobQueue.size()
            << std::endl;

    // Thread creation
    constexpr int WORKER_COUNT = 4;
    pthread_t workerThreads[WORKER_COUNT];
    pthread_t fetcherThread;
    pthread_t responderThread;

    double startTime = MPI_Wtime();

    pthread_create(
            &responderThread,
            nullptr,
            responder,
            nullptr
    );

    for (int i = 0; i < WORKER_COUNT; ++i) {
        pthread_create(
                &workerThreads[i],
                nullptr,
                worker,
                nullptr
        );
    }

    pthread_create(
            &fetcherThread,
            nullptr,
            fetcher,
            nullptr
    );

    // Thread synchronization
    for (int i = 0; i < WORKER_COUNT; ++i) {
        pthread_join(workerThreads[i], nullptr);
    }

    pthread_join(fetcherThread, nullptr);
    pthread_join(responderThread, nullptr);

    double localTime = MPI_Wtime() - startTime;

    // Statistics
    int localCompleted = jobsCompleted.load();
    int localSent = jobsSent.load();
    int localReceived = jobsReceived.load();

    int globalCompleted = 0;
    int globalSent = 0;
    int globalReceived = 0;

    MPI_Reduce(
            &localCompleted,
            &globalCompleted,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
    );

    MPI_Reduce(
            &localSent,
            &globalSent,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
    );

    MPI_Reduce(
            &localReceived,
            &globalReceived,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
    );

    double maxTime = 0.0;

    MPI_Reduce(
            &localTime,
            &maxTime,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
    );
    std::cout
            << "[Rank " << rank << "] "
            << "Completed: " << localCompleted
            << ", Sent: " << localSent
            << ", Received: " << localReceived
            << std::endl;

    if (rank == 0) {
        std::cout
                << "\n========== DWSP RESULTS =========="
                << std::endl;

        std::cout
                << "MPI processes: " << size
                << std::endl;

        std::cout
                << "Total jobs: " << totalJobs
                << std::endl;

        std::cout
                << "Completed jobs: " << globalCompleted
                << std::endl;

        std::cout
                << "Transferred jobs: " << globalSent
                << std::endl;

        std::cout
                << "Received jobs: " << globalReceived
                << std::endl;

        std::cout
                << "Execution time: " << maxTime
                << " seconds"
                << std::endl;

        std::cout
                << "=================================="
                << std::endl;
    }
    pthread_cond_destroy(&c_jobAdded);
    pthread_cond_destroy(&c_needToFetch);
    pthread_mutex_destroy(&mutex);
    MPI_Comm_free(&replyComm);
    MPI_Finalize();
    return 0;
}