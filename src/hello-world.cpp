#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int processCount = 0, processId = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    MPI_Finalize();
    return 0;
}
