#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std::chrono;

void matrix_vector_multiply(const double *mat, const double *vec, int N, double *new_vector) {
    for (int i = 0; i < N; i++) {
        new_vector[i] = 0;
        for (int j = 0; j < N; j++) {
            new_vector[i] += mat[i * N + j] * vec[j];
        }
    }
}

void multiply_by_const(const double *vec, double c, int size, double *new_vector) {
    for (int i = 0; i < size; i++) {
        new_vector[i] = vec[i] * c;
    }
}

void vector_subtraction(const double *vec1, const double *vec2, int size, double *new_vector) {
    for (int i = 0; i < size; i++) {
        new_vector[i] = vec1[i] - vec2[i];
    }
}

void vector_summation(const double *vec1, const double *vec2, int size, double *new_vector) {
    for (int i = 0; i < size; i++) {
        new_vector[i] = vec1[i] + vec2[i];
    }
}

double dot_product(const double *vec1, const double *vec2, int size) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

void print_matrix(double *mat, int rows, int columns, std::ofstream &stream) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            stream << mat[i * columns + j] << " ";
        }
        stream << std::endl;
    }
}

double *solve_SLAE(const double *A, double *b, int N) {
    auto *solution = new double[N]; //xn+1
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

    double norm_b = sqrt(dot_product(b, b, N));
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
        matrix_vector_multiply(A, prev_solution, N, tmp_A);
        vector_subtraction(b, tmp_A, N, r);
        /// zn = rn
        for (int i = 0; i < N; ++i) {
            z[i] = r[i];
        }
        /// alpha
        matrix_vector_multiply(A, z, N, tmp_A);
        alpha = dot_product(r, r, N) / dot_product(tmp_A, z, N);
        /// xn+1 = xn + alpha * zn
        multiply_by_const(z, alpha, N, alpha_z);
        vector_summation(prev_solution, alpha_z, N, solution);
        /// rn+1 = rn - alpha * A * zn
        matrix_vector_multiply(A, alpha_z, N, tmp_A);
        vector_subtraction(r, tmp_A, N, r_next);
        /// beta
        beta = dot_product(r_next, r_next, N) / dot_product(r, r, N);
        /// zn+1 = rn+1 + beta * zn
        multiply_by_const(z, beta, N, beta_z);
        vector_summation(r_next, beta_z, N, z_next);

        res = sqrt(dot_product(r, r, N)) / norm_b;
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

    std::cout << "iter_count: " << iter_count << std::endl;

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

    const std::string &file_name = name;
    std::ofstream file_stream(file_name);
    if (!file_stream) {
        std::cout << "error with output file" << std::endl;
        return 0;
    }

    file_stream << "Matrix size: " << N << std::endl;

    auto *b = new double[N];
    auto *u = new double[N];
    auto *A = new double[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[i * N + j] = 2;
            } else {
                A[i * N + j] = 1;
            }
        }
        u[i] = sin(2 * M_PI * i / double(N));
    }
    matrix_vector_multiply(A, u, N, b);

    auto start_time = system_clock::now();
    double *solution = solve_SLAE(A, b, N);
    auto end_time = system_clock::now();
    auto duration = duration_cast<nanoseconds>(end_time - start_time);

    if (solution != nullptr) {
        file_stream << "Answer:" << std::endl;
        print_matrix(u, 1, N, file_stream);
        file_stream << "SLAE solution:" << std::endl;
        print_matrix(solution, 1, N, file_stream);
        file_stream << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
    } else {
        file_stream << "Does not converge" << std::endl;
    }

    delete[](solution);
    delete[](b);
    delete[](u);
    delete[](A);
    return 0;
}
