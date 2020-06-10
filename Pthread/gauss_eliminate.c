/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student names(s): Daniel Drzewicki, Brian Tu
 * Date: 2/6/2020
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 4

typedef struct thread_data_s
{
	int tid;
	unsigned int num_elements;
	int num_threads;
	float* U;
} thread_data_t;

pthread_barrier_t phase1_barrier;
pthread_barrier_t row_barrier;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.6f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    gettimeofday(&start, NULL);
    
    gauss_eliminate_using_pthreads(U_mt);

    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.6f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* Worker function */
void* threading(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
    float* U = thread_data->U;
    int num_elements = thread_data->num_elements;
    
    int i, j, k, l;
    for (k = 0; k < num_elements; k++) {
        if ((k % thread_data->num_threads) == thread_data->tid) {
            for (j = k + 1; j < num_elements; j++) {
                U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);
            }
            U[num_elements * k + k] = 1;
        }
        pthread_barrier_wait(&row_barrier);
        for (i = (k + 1); i < num_elements; i++) {
            if ((i % thread_data->num_threads) == thread_data->tid) {
                for (l = k + 1; l < num_elements; l++) {
                    U[num_elements * i + l] = (float)(U[num_elements * i + l] - U[num_elements * k + l] * U[num_elements * i + k]);
                }
                U[num_elements * i + k] = 0;
            }
        }
    }
    if (thread_data->tid == thread_data->num_threads - 1) {
        U[(num_elements - 1) * num_elements + num_elements - 1] = 1;
    }
    pthread_exit(NULL);
}


void gauss_eliminate_using_pthreads(Matrix U)
{
    pthread_t *thread_id = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));    /* Data structure to store thread IDs */
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * NUM_THREADS);
    pthread_attr_t attributes;
    pthread_attr_init (&attributes);
    pthread_barrier_init(&row_barrier, NULL, NUM_THREADS);
    
    int i;
    for (i = 0; i < NUM_THREADS; i++) {
        thread_data[i].tid          = i;
        thread_data[i].num_threads  = NUM_THREADS;
        thread_data[i].num_elements = U.num_rows;
        thread_data[i].U            = U.elements;
        pthread_create(&thread_id[i], &attributes, threading, (void *)&thread_data[i]);
    }

    /* Wait for worker threads to finish */
    for (i = 0; i < NUM_THREADS; i++)
        pthread_join(thread_id[i], NULL);

    pthread_barrier_destroy(&row_barrier);

    /* Free data structures */
    free((void *)thread_data);
    free((void *)thread_id);
}


/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}
