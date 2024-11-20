#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Win win;
    long long int *total_count;
    MPI_Win_allocate(sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, total_count, &win);
    if (world_rank == 0) {
        *total_count = 0;
    }

    MPI_Win_fence(0, win);

    // TODO: MPI init
    
    long long int count = 0;
    long long int total_iter;
    unsigned int seed = world_rank;

    if (world_rank > 0)
        total_iter = tosses / world_size;
    else if (world_rank == 0)
        total_iter = tosses / world_size + tosses % world_size;

    for (long long int i = 0; i < total_iter; i++){
        double temp1 = rand_r(&seed) / (double)RAND_MAX; 
        double temp2 = rand_r(&seed) / (double)RAND_MAX; 
        if (temp1 * temp1 + temp2 * temp2 <= 1.0)
            count++;
    }

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    *total_count += count;
    MPI_Win_unlock(0, win);

    MPI_Win_fence(0, win);

    if (world_rank == 0){
        count = *total_count;
    }
    

    // if (world_rank == 0)
    // {
    //     // Master
    // }
    // else
    // {
    //     // Workers
    // }


    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 / (double)tosses * (double)count;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}