#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// ----------------- Standard DFT -----------------

void dft(const double complex *input, double complex *output, int N) {
    for (int k=0; k<N; ++k) {
        output[k] = 0.0 + 0.0*I;
        for (int n=0; n<N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            output[k] += input[n] * cexp(I * angle);
        }
    }
}

// ----------------- Dihedral D_n FFT Optimized -----------------

#define MATRIX_SIZE 4  // 2x2 matrix flattened

typedef struct {
    int n;
    double complex* rho_k_r_cache; // size n*n*4
    double complex* rho_k_s_cache; // size n*4
} DihedralCache;

void precompute_dihedral_rho(DihedralCache* cache) {
    int n = cache->n;
    cache->rho_k_r_cache = malloc(sizeof(double complex) * n * n * MATRIX_SIZE);
    cache->rho_k_s_cache = malloc(sizeof(double complex) * n * MATRIX_SIZE);

    for (int k=0; k<n; k++) {
        for (int m=0; m<n; m++) {
            double angle = 2.0 * M_PI * k * m / n;
            int base = (k * n + m) * MATRIX_SIZE;
            cache->rho_k_r_cache[base + 0] = cos(angle) + 0.0*I;
            cache->rho_k_r_cache[base + 1] = -sin(angle) + 0.0*I;
            cache->rho_k_r_cache[base + 2] = sin(angle) + 0.0*I;
            cache->rho_k_r_cache[base + 3] = cos(angle) + 0.0*I;
        }
        int s_base = k * MATRIX_SIZE;
        cache->rho_k_s_cache[s_base + 0] = 1.0 + 0.0*I;
        cache->rho_k_s_cache[s_base + 1] = 0.0 + 0.0*I;
        cache->rho_k_s_cache[s_base + 2] = 0.0 + 0.0*I;
        cache->rho_k_s_cache[s_base + 3] = -1.0 + 0.0*I;
    }
}

static inline void mat_mul_2x2_opt(const double complex* A, const double complex* B, double complex* C) {
    C[0] = A[0] * B[0] + A[1] * B[2];
    C[1] = A[0] * B[1] + A[1] * B[3];
    C[2] = A[2] * B[0] + A[3] * B[2];
    C[3] = A[2] * B[1] + A[3] * B[3];
}

void dn_fft_2d_irrep_opt(const double complex* input, double complex* output, int n, int k, DihedralCache* cache) {
    for (int i = 0; i < MATRIX_SIZE; i++) output[i] = 0.0 + 0.0*I;

    #pragma omp parallel
    {
        double complex sum_local[MATRIX_SIZE] = {0};
        #pragma omp for nowait
        for (int g = 0; g < 2 * n; g++) {
            int m = g % n;
            int refl = g / n;

            double complex rho_r[4];
            double complex rho_g[4];
            int rot_index = (k * n + m) * MATRIX_SIZE;
            for (int i = 0; i < MATRIX_SIZE; i++) {
                rho_r[i] = cache->rho_k_r_cache[rot_index + i];
            }

            if (refl == 0) {
                for (int i = 0; i < MATRIX_SIZE; i++) rho_g[i] = rho_r[i];
            }
            else {
                double complex temp[4];
                int s_index = k * MATRIX_SIZE;
                for (int i = 0; i < MATRIX_SIZE; i++) temp[i] = cache->rho_k_s_cache[s_index + i];
                mat_mul_2x2_opt(temp, rho_r, rho_g);
            }

            double complex val = input[g];
            for (int i = 0; i < MATRIX_SIZE; i++) sum_local[i] += val * rho_g[i];
        }
        #pragma omp critical
        {
            for (int i = 0; i < MATRIX_SIZE; i++)
                output[i] += sum_local[i];
        }
    }
}

// ----------------- Quaternion Q8 FFT Optimized -----------------

static const double complex q8_2d_irrep_mats[8][4] = {
    {1, 0, 0, 1}, {-1, 0, 0, -1}, {0, 1, 1, 0}, {0, -1, -1, 0},
    {0, -I, I, 0}, {0, I, -I, 0}, {1, 0, 0, -1}, {-1, 0, 0, 1}
};

void q8_fft_2d_irrep_opt(const double complex* input, double complex* output) {
    for (int i = 0; i < MATRIX_SIZE; i++) output[i] = 0.0 + 0.0 * I;

    #pragma omp parallel
    {
        double complex sum_local[MATRIX_SIZE] = {0};
        #pragma omp for nowait
        for (int g = 0; g < 8; g++) {
            double complex val = input[g];
            if (cabs(val) == 0.0) continue;
            for (int i = 0; i < MATRIX_SIZE; i++) sum_local[i] += val * q8_2d_irrep_mats[g][i];
        }
        #pragma omp critical
        {
            for (int i = 0; i < MATRIX_SIZE; i++)
                output[i] += sum_local[i];
        }
    }
}

// ================== BENCHMARK HELPERS ==================

void benchmark_dft(const double complex* input, int N, double complex* output) {
    clock_t start = clock();
    dft(input, output, N);
    clock_t end = clock();
    printf("Standard DFT took %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}

void benchmark_dn_fft_2d(const double complex* input, int n, DihedralCache* cache, double complex* output) {
    clock_t start = clock();
    dn_fft_2d_irrep_opt(input, output, n, 1, cache);
    clock_t end = clock();
    printf("Dihedral FFT (2D irrep k=1) took %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}

void benchmark_q8_fft_2d(const double complex* input, double complex* output) {
    clock_t start = clock();
    q8_fft_2d_irrep_opt(input, output);
    clock_t end = clock();
    printf("Quaternion Q8 FFT (2D irrep) took %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}

// ----------------- MAIN -----------------

int main() {
    int n = 4;
    int N_dn = 2 * n;
    double complex input_dn[8];
    double complex input_q8[8];
    
    // Seed random number generator for different results each run
    srand((unsigned)time(NULL));
    
    // Generate random complex input signals for dihedral D_n and quaternion Q_8
    for (int i = 0; i < N_dn; ++i) {
        double real_part = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // [-1, 1]
        double imag_part = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        input_dn[i] = real_part + I * imag_part;
    }
    for (int i = 0; i < 8; ++i) {
        double real_part = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double imag_part = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        input_q8[i] = real_part + I * imag_part;
    }

    // Buffers for outputs
    double complex dft_out[8];
    double complex dn_fft_out[4];
    double complex q8_fft_out[4];

    // Benchmark Standard DFT (size = 8)
    benchmark_dft(input_dn, 8, dft_out);

    // Precompute dihedral representation matrices
    DihedralCache dn_cache = { n, NULL, NULL };
    precompute_dihedral_rho(&dn_cache);

    // Benchmark Dihedral FFT 2D irrep (k=1)
    benchmark_dn_fft_2d(input_dn, n, &dn_cache, dn_fft_out);

    // Benchmark Quaternion Q8 FFT 2D irrep
    benchmark_q8_fft_2d(input_q8, q8_fft_out);

    // Cleanup
    free(dn_cache.rho_k_r_cache);
    free(dn_cache.rho_k_s_cache);

    return 0;
}
