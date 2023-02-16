#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include "mpi.h"

using namespace std::chrono;

void
matrix_vector_multiply(const double *mat, const double *vec, int *sizes_per_threads, int *dispositions, int N, double *new_vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *tmp_vector = new double[sizes_per_threads[rank]];
    for (int i = 0; i < sizes_per_threads[rank]; i++) {
        tmp_vector[i] = 0;
        for (int j = 0; j < N; j++) {
            tmp_vector[i] += mat[i * N + j] * vec[j];
        }
    }

    MPI_Allgatherv(tmp_vector, sizes_per_threads[rank], MPI_DOUBLE, new_vector,
                   sizes_per_threads, dispositions, MPI_DOUBLE, MPI_COMM_WORLD);
    delete[](tmp_vector);
}

void
multiply_by_const(const double *vec, double c, const int *sizes_per_threads, const int *dispositions, int N, double *new_vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto *tmp_vector = new double[N];
    std::fill(tmp_vector, tmp_vector + N, 0);
    for (int i = dispositions[rank]; i < dispositions[rank] + sizes_per_threads[rank]; i++) {
        tmp_vector[i] = vec[i] * c;
    }
    MPI_Allreduce(tmp_vector, new_vector, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](tmp_vector);
}

void
vector_subtraction(const double *vec1, const double *vec2, const int *sizes_per_threads, const int *dispositions, int N,
       double *new_vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto *tmp_vector = new double[N];
    std::fill(tmp_vector, tmp_vector + N, 0);
    for (int i = dispositions[rank]; i < dispositions[rank] + sizes_per_threads[rank]; i++) {
        tmp_vector[i] = vec1[i] - vec2[i];
    }
    MPI_Allreduce(tmp_vector, new_vector, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](tmp_vector);
}

void
vector_summation(const double *vec1, const double *vec2, const int *sizes_per_threads, const int *dispositions, int N,
       double *new_vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto *tmp_vector = new double[N];
    std::fill(tmp_vector, tmp_vector + N, 0);
    for (int i = dispositions[rank]; i < dispositions[rank] + sizes_per_threads[rank]; i++) {
        tmp_vector[i] = vec1[i] + vec2[i];
    }
    MPI_Allreduce(tmp_vector, new_vector, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](tmp_vector);
}

double dot_product(const double *vec1, const double *vec2, const int *sizes_per_threads, const int *dispositions) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sum = 0;
    for (int i = dispositions[rank]; i < dispositions[rank] + sizes_per_threads[rank]; i++) {
        sum += vec1[i] * vec2[i];
    }
    double full_sum;
    MPI_Allreduce(&sum, &full_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return full_sum;
}

void print_matrix(double *mat, int rows, int columns, std::ofstream &stream) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            stream << mat[i * columns + j] << " ";
        }
        stream << std::endl;
    }
}

double *solve_SLAE(const double *A, double *b, int *sizes_per_threads, int *dispositions, int N) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *solution = new double[N]; // xn+1
    std::fill(solution, solution + N, 0);
    auto *prev_solution = new double[N]; // xn
    std::fill(prev_solution, prev_solution + N, 0);

    auto *tmp_A = new double[N];
    auto *r = new double[N];
    auto *z = new double[N];
    auto *r_next = new double[N];
    auto *z_next = new double[N];
    auto *alpha_z = new double[N];
    auto *beta_z = new double[N];

    double alpha;
    double beta;

    const double EPSILON = 1e-009;

    double norm_b = sqrt(dot_product(b, b, sizes_per_threads, dispositions));
    double dot_rr;

    double res = 1;
    double prev_res = 1;
    bool diverge = false;
    int diverge_count = 0;
    int right_answer_repeat = 0;
    int iter_count = 0;
    while (res > EPSILON || right_answer_repeat < 5) {
        if (res < EPSILON) {
            ++right_answer_repeat;
        } else {
            right_answer_repeat = 0;
        }

        /// rn = b - A * xn
        matrix_vector_multiply(A, prev_solution, sizes_per_threads, dispositions, N, tmp_A);
        vector_subtraction(b, tmp_A, sizes_per_threads, dispositions, N, r);
        /// zn = rn
        for (int i = 0; i < N; ++i) {
            z[i] = r[i];
        }
        /// alpha
        matrix_vector_multiply(A, z, sizes_per_threads, dispositions, N, tmp_A);
        dot_rr = dot_product(r, r, sizes_per_threads, dispositions);
        alpha = dot_rr / dot_product(tmp_A, z, sizes_per_threads, dispositions);
        /// xn+1 = xn + alpha * zn
        multiply_by_const(z, alpha, sizes_per_threads, dispositions, N, alpha_z);
        vector_summation(prev_solution, alpha_z, sizes_per_threads, dispositions, N, solution);
        /// rn+1 = rn - alpha * A * zn
        matrix_vector_multiply(A, alpha_z, sizes_per_threads, dispositions, N, tmp_A);
        vector_subtraction(r, tmp_A, sizes_per_threads, dispositions, N, r_next);
        /// beta
        beta = dot_product(r_next, r_next, sizes_per_threads, dispositions) / dot_rr;
        /// zn+1 = rn+1 + beta * zn
        multiply_by_const(z, beta, sizes_per_threads, dispositions, N, beta_z);
        vector_summation(r_next, beta_z, sizes_per_threads, dispositions, N, z_next);

        res = sqrt(dot_rr) / norm_b;
        if (prev_res < res || res == INFINITY || res == NAN) {
            ++diverge_count;
            if (diverge_count > 10 || res == INFINITY || res == NAN) {
                diverge = true;
                break;
            }
        } else {
            diverge_count = 0;
        }
        prev_res = res;
        for (long i = 0; i < N; i++) {
            prev_solution[i] = solution[i];
            r[i] = r_next[i];
            z[i] = z_next[i];
        }
        ++iter_count;
    }
    delete[](prev_solution);
    delete[](tmp_A);
    delete[](r);
    delete[](z);
    delete[](r_next);
    delete[](z_next);
    delete[](alpha_z);
    delete[](beta_z);

    std::cout << "iterCount: " << iter_count << std::endl;

    if (diverge) {
        delete[](solution);
        return nullptr;
    } else {
        return solution;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Program needs 2 arguments: size, filename" << std::endl;
        return 0;
    }
    int N = atoi(argv[1]);
    std::string name(argv[2]);

    MPI_Init(&argc, &argv);
    int thread_count, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &thread_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string file_name = std::to_string(rank) + "rank-" + name;
    std::ofstream file_stream(file_name);
    if (!file_stream) {
        std::cout << "error with output file" << std::endl;
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        file_stream << "Matrix size: " << N << " threadCount: " << thread_count << std::endl;
    }

    int sizes_per_threads[thread_count];
    int dispositions[thread_count];
    std::fill(sizes_per_threads, sizes_per_threads + thread_count, N / thread_count);
    sizes_per_threads[thread_count - 1] += N % thread_count;
    dispositions[0] = 0;
    for (int i = 1; i < thread_count; ++i) {
        dispositions[i] = dispositions[i - 1] + sizes_per_threads[i - 1];
    }

    auto *b = new double[N];
    auto *u = new double[N];
    auto *A = new double[sizes_per_threads[rank] * N];

    for (int i = 0; i < sizes_per_threads[rank]; i++) {
        for (int j = 0; j < N; j++) {
            if (i + dispositions[rank] == j) {
                A[i * N + j] = 2;
            } else {
                A[i * N + j] = 1;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        u[i] = sin(2 * M_PI * i / double(N));
    }
    matrix_vector_multiply(A, u, sizes_per_threads, dispositions, N, b);

    auto start_time = system_clock::now();
    double *solution = solve_SLAE(A, b, sizes_per_threads, dispositions, N);
    auto end_time = system_clock::now();
    auto duration = duration_cast<nanoseconds>(end_time - start_time);

    if (rank == 0 && solution != nullptr) {
        file_stream << "Answer:" << std::endl;
        print_matrix(u, 1, N, file_stream);
        file_stream << "SLAE solution:" << std::endl;
        print_matrix(solution, 1, N, file_stream);
        file_stream << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
    } else if (solution == nullptr) {
        file_stream << "Does not converge" << std::endl;
    }

    delete[](solution);
    delete[](b);
    delete[](u);
    delete[](A);
    MPI_Finalize();
    return 0;
}
