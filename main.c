//Matrix Multiplication with OpenMP

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define EPSILON 0.000001
#define N 1000
#define BLOCK_SIZE 64
#define min(a,b) (((a)<(b))?(a):(b))
//#define DEBUG

//MATRICE
double a[N][N], b[N][N], c[N][N], c2[N][N];
double ijk[N][N];

void Generate_matrix(char *prompt, double mat[N][N])
{
    int i, j;
    printf("%s\n", prompt);
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            mat[i][j] = (double)rand()/RAND_MAX;
}

void Print_matrix(char *title, double mat[N][N])
{
    int i, j;
    printf("%s\n", title);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            printf("%4.1f ", mat[i][j]);
        printf("\n");
    }
}

int Equal_matrixes(double mat1[N][N], double mat2[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            if (fabs(mat1[i][j] - mat2[i][j]) > EPSILON)
            {
                return 0;
            }
    }
    return 1;
}

//INMULTIRE SERIAL
void serial_ijk()
{
    int i, j, k;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
        }
}
void serial_ikj()
{
    int i, j, k;
    double aik;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0;

    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++)
        {
            aik = a[i][k];
            for (int j = 0; j < N; j++)
            {
                c[i][j] += aik * b[k][j];
            }
        }
}
void serial_jik()
{
    int i, j, k;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0;

    for (j = 0; j < N; j++)
        for (i = 0; i < N; i++)
        {
            double suma=0.0;
            for (k = 0; k < N; k++)
                suma+=a[i][k]*b[k][j];
            c[i][j] = suma;
        }
}
void serial_jki()
{
    int i, j, k;
    double bkj;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0;

    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
        {
            bkj = b[k][j];
            for (int i = 0; i < N; i++)
            {
                c[i][j] += a[i][k] * bkj;
            }
        }
}
void serial_kij()
{
    int i, j, k;
    double aik;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0;

    for (k = 0; k < N; k++)
        for (i = 0; i < N; i++)
        {
            aik = a[i][k];
            for (j = 0; j < N; j++)
                c[i][j] += aik * b[k][j];
        }
}

void serial_kji()
{
    int i, j, k;
    double bkj;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0;

    for (k = 0; k < N; k++)
    {
        for (j = 0; j < N; j++)
        {
            bkj = b[k][j];
            for (i = 0; i < N; i++)
            {
                c[i][j] += a[i][k] * bkj;
            }
        }
    }
}

//INMULTIRE PARALLEL
void parallel_ijk(int nthreads, int chunk)
{
    int i, j, k;
    double temp;
#pragma omp parallel num_threads(nthreads), default(none), private(i, j, k, temp), shared(a, b, c2, chunk)
    {
#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                c2[i][j] = 0;

#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
            {
                for (k = 0; k < N; k++)
                    c2[i][j] += a[i][k] * b[k][j];
            }
    }
}
void parallel_ikj(int nthreads, int chunk)
{
    int i, j, k;
    double aik;
#pragma omp parallel num_threads(nthreads), default(none), private(i, j, k, aik), shared(a, b, c2, chunk)
    {
#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                c2[i][j] = 0;

#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (k = 0; k < N; k++)
            {
                aik = a[i][k];
                for (int j = 0; j < N; j++)
                {
                    c2[i][j] += aik * b[k][j];
                }
            }
    }
}
void parallel_jik(int nthreads, int chunk)
{
    int i, j, k;

#pragma omp parallel num_threads(nthreads), default(none), private(i, j, k), shared(a, b, c2, chunk)
    {
#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                c2[i][j] = 0;

#pragma omp for schedule(static, chunk)
        for (j = 0; j < N; j++)
        {
            for (i = 0; i < N; i++)
            {
                double sum = 0.0;
                for (k = 0; k < N; k++)
                {
                    sum += a[i][k] * b[k][j];
                }
                c2[i][j] = sum;
            }
        }
    }
}
void parallel_jki(int nthreads, int chunk)
{
    int i, j, k;
    double bkj;

#pragma omp parallel num_threads(nthreads), default(none), private(i, j, k, bkj), shared(a, b, c2, chunk)
    {
#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                c2[i][j] = 0.0;

#pragma omp for schedule(static, chunk)
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                bkj = b[k][j];
                for (i = 0; i < N; i++)
                {
                    c2[i][j] += a[i][k] * bkj;
                }
            }
        }
    }
}
void parallel_kij(int nthreads, int chunk)
{
    int i, j, k;
    double aik;

#pragma omp parallel num_threads(nthreads), default(none), private(i, j, k, aik), shared(a, b, c2, chunk)
    {
#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                c2[i][j] = 0.0;

        for (k = 0; k < N; k++)
        {
#pragma omp for schedule(static, chunk)
            for (i = 0; i < N; i++)
            {
                aik = a[i][k];
                for (j = 0; j < N; j++)
                {
                    c2[i][j] += aik * b[k][j];
                }
            }
        }
    }
}
void parallel_kji(int nthreads, int chunk)
{
    int i, j, k;
    double bkj;

#pragma omp parallel num_threads(nthreads), default(none), private(i, j, k, bkj), shared(a, b, c2, chunk)
    {
#pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                c2[i][j] = 0.0;

        for (k = 0; k < N; k++)
        {
#pragma omp for schedule(static, chunk)
            for (j = 0; j < N; j++)
            {
                bkj = b[k][j];
                for (i = 0; i < N; i++)
                {
                    c2[i][j] += a[i][k] * bkj;
                }
            }
        }
    }
}

//INMULTIRE PE BLOCURI SERIAL
void serial_blocked(int B)
{
    int i, j, k, ii, jj, kk;
    int i_end, j_end, k_end;
    double aik;

    for (ii = 0; ii < N; ii += B)
    {
        for (jj = 0; jj < N; jj += B)
        {
            i_end = min(ii + B, N);
            j_end = min(jj + B, N);
            for (i = ii; i < i_end; i++)
            {
                for (j = jj; j < j_end; j++)
                {
                    c[i][j] = 0.0;
                }
            }

            for (kk = 0; kk < N; kk += B)
            {
                k_end = min(kk + B, N);
                for (i = ii; i < i_end; i++)
                {
                    for (k = kk; k < k_end; k++)
                    {
                        aik = a[i][k];
                        for (j = jj; j < j_end; j++)
                        {
                            c[i][j] += aik * b[k][j];
                        }
                    }
                }
            }
        }
    }
}

//INMULTIRE PE BLOCURI PARALEL
void parallel_blocked(int B, int nthreads)
{
    int i, j, k, ii, jj, kk;
    int i_end, j_end, k_end;
    double aik;

#pragma omp parallel for num_threads(nthreads) private(j) schedule(static)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            c2[i][j] = 0.0;
        }
    }

#pragma omp parallel num_threads(nthreads), default(none), shared(a, b, c2, B, nthreads) private(i, j, k, ii, jj, kk, i_end, j_end, k_end, aik)
    {
#pragma omp for schedule(dynamic)
        for (ii = 0; ii < N; ii += B)
        {
            for (jj = 0; jj < N; jj += B)
            {
                i_end = min(ii + B, N);
                j_end = min(jj + B, N);
                for (kk = 0; kk < N; kk += B)
                {
                    k_end = min(kk + B, N);
                    for (i = ii; i < i_end; i++)
                    {
                        for (k = kk; k < k_end; k++)
                        {
                            aik = a[i][k];
                            for (j = jj; j < j_end; j++)
                            {
                                c2[i][j] += aik * b[k][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

//MAIN
int main(void)
{
    int nthreads, chunk;
    double start, end, time_serial, time_parallel;
    nthreads=8;
    chunk=10;

    //Generare matrice
    srand(time(NULL));
    Generate_matrix("Generating matrix a ...", a);
    Generate_matrix("Generating matrix b ...", b);

//IJK
    printf("\n- IJK - \n");
    //Serial
    printf("Start working serial ijk ... \n");
    start = omp_get_wtime();
    serial_ijk();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time ijk %lf seconds \n", time_serial);

    //copiere in ijk
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            ijk[i][j] = c[i][j];

    if (!Equal_matrixes(ijk, c))
        printf("Attention! Serial ijk Result not the same as Gold! \n");

#ifdef DEBUG
    Print_matrix("Serial result ijk...", c);
#endif

    //Parallel
    printf("Start working parallel ijk with %d threads ... \n", nthreads);
    start = omp_get_wtime();
    parallel_ijk(nthreads, chunk);
    end = omp_get_wtime();
    time_parallel = (end - start);
    printf("Parallel time ijk %lf seconds \n", time_parallel);

#ifdef DEBUG
    Print_matrix("Parallel result ijk...", c2);
#endif

    //Speedup
    printf("Speedup = %2.2lf\n", time_serial / time_parallel);
    if (!Equal_matrixes(ijk, c2))
        printf("Attention! Serial and Parallel ijk Result not the same ! \n");

//IKJ
    printf("\n- IKJ -\n");
    //Serial
    printf("Start working serial ikj ... \n");
    start = omp_get_wtime();
    serial_ikj();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time ikj %lf seconds \n", time_serial);

    if (!Equal_matrixes(ijk, c))
        printf("Attention! Serial ikj Result not the same as Gold! \n");

#ifdef DEBUG
    Print_matrix("Serial result ikj...", c);
#endif

    //Parallel
    printf("Start working parallel ikj with %d threads ... \n", nthreads);
    start = omp_get_wtime();
    parallel_ikj(nthreads, chunk);
    end = omp_get_wtime();
    time_parallel = (end - start);
    printf("Parallel time ikj %lf seconds \n", time_parallel);

#ifdef DEBUG
    Print_matrix("Parallel result ikj...", c2);
#endif

    //Speedup
    printf("Speedup = %2.2lf\n", time_serial / time_parallel);
    if (!Equal_matrixes(ijk, c2))
        printf("Attention! Serial and Parallel ikj Result not the same ! \n");

//JIK
    printf("\n- JIK -\n");
    //Serial
    printf("Start working serial jik ... \n");
    start = omp_get_wtime();
    serial_jik();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time jik %lf seconds \n", time_serial);

    if (!Equal_matrixes(ijk, c))
        printf("Attention! Serial jik Result not the same as Gold! \n");

#ifdef DEBUG
    Print_matrix("Serial result jik...", c);
#endif

    //Parallel
    printf("Start working parallel jik with %d threads ... \n", nthreads);
    start = omp_get_wtime();
    parallel_jik(nthreads, chunk);
    end = omp_get_wtime();
    time_parallel = (end - start);
    printf("Parallel time jik %lf seconds \n", time_parallel);

#ifdef DEBUG
    Print_matrix("Parallel result jik...", c2);
#endif

    //Speedup
    printf("Speedup = %2.2lf\n", time_serial / time_parallel);
    if (!Equal_matrixes(ijk, c2))
        printf("Attention! Serial and Parallel jik Result not the same ! \n");

//JKI
    printf("\n- JKI -\n");
    //Serial
    printf("Start working serial jki ... \n");
    start = omp_get_wtime();
    serial_jki();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time jki %lf seconds \n", time_serial);

    if (!Equal_matrixes(ijk, c))
        printf("Attention! Serial jki Result not the same as Gold! \n");

#ifdef DEBUG
    Print_matrix("Serial result jki...", c);
#endif

    //Parallel
    printf("Start working parallel jki with %d threads ... \n", nthreads);
    start = omp_get_wtime();
    parallel_jki(nthreads, chunk);
    end = omp_get_wtime();
    time_parallel = (end - start);
    printf("Parallel time jki %lf seconds \n", time_parallel);

#ifdef DEBUG
    Print_matrix("Parallel result jki...", c2);
#endif

    //Speedup
    printf("Speedup = %2.2lf\n", time_serial / time_parallel);
    if (!Equal_matrixes(ijk, c2))
        printf("Attention! Serial and Parallel jki Result not the same ! \n");

//KIJ
    printf("\n- KIJ -\n");
    //Serial
    printf("Start working serial kij ... \n");
    start = omp_get_wtime();
    serial_kij();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time kij %lf seconds \n", time_serial);

    if (!Equal_matrixes(ijk, c))
        printf("Attention! Serial kij Result not the same as Gold! \n");

#ifdef DEBUG
    Print_matrix("Serial result kij...", c);
#endif

    //Parallel
    printf("Start working parallel kij with %d threads ... \n", nthreads);
    start = omp_get_wtime();
    parallel_kij(nthreads, chunk);
    end = omp_get_wtime();
    time_parallel = (end - start);
    printf("Parallel time kij %lf seconds \n", time_parallel);

#ifdef DEBUG
    Print_matrix("Parallel result kij...", c2);
#endif

    //Speedup
    printf("Speedup = %2.2lf\n", time_serial / time_parallel);
    if (!Equal_matrixes(ijk, c2))
        printf("Attention! Serial and Parallel kij Result not the same ! \n");

//KJI
    printf("\n- KJI -\n");
    //Serial
    printf("Start working serial kji ... \n");
    start = omp_get_wtime();
    serial_kji();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time kji %lf seconds \n", time_serial);

    if (!Equal_matrixes(ijk, c))
        printf("Attention! Serial kji Result not the same as Gold! \n");

#ifdef DEBUG
    Print_matrix("Serial result kji...", c);
#endif

    //Parallel
    printf("Start working parallel kji with %d threads ... \n", nthreads);
    start = omp_get_wtime();
    parallel_kji(nthreads, chunk);
    end = omp_get_wtime();
    time_parallel = (end - start);
    printf("Parallel time kji %lf seconds \n", time_parallel);

#ifdef DEBUG
    Print_matrix("Parallel result kji...", c2);
#endif

    //Speedup
    printf("Speedup = %2.2lf\n", time_serial / time_parallel);
    if (!Equal_matrixes(ijk, c2))
        printf("Attention! Serial and Parallel kji Result not the same ! \n");

//BLOCKED
    int b=BLOCK_SIZE;
    printf("\n- BLOCKED (Block Size = %d) -\n", b);
    //Serial
    printf("Start working serial blocked ... \n");
    start = omp_get_wtime();
    serial_blocked(b);
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time blocked %lf seconds \n", time_serial);

    if (!Equal_matrixes(ijk, c))
        printf("Attention! Serial blocked Result not the same as Gold! \n");

#ifdef DEBUG
    Print_matrix("Parallel result kji...", c2);
#endif

    // Parallel
    printf("Start working parallel blocked with %d threads ... \n", nthreads);
    start = omp_get_wtime();
    parallel_blocked(b, nthreads);
    end = omp_get_wtime();
    time_parallel = (end - start);
    printf("Parallel time blocked %lf seconds \n", time_parallel);

#ifdef DEBUG
    Print_matrix("Parallel result kji...", c2);
#endif

    //Speedup
    printf("Speedup = %2.2lf\n", time_serial / time_parallel);
    if (!Equal_matrixes(ijk, c2))
        printf("Attention! Parallel blocked Result not the same as Gold! \n");

    return 0;
}