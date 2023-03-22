#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int processCount = 0, processId = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    int iterations = 10000000;
    double local = 0.0;

    double startTime = MPI_Wtime();

    for (int i = processId; i < iterations; i += processCount)
    {
        double x = (i + 0.5) / iterations;
        local += 4.0 / (1.0 + x * x);
    }
    local /= iterations;

    if (processId != 0)
    {
        MPI_Send(&local, 1, MPI_DOUBLE, 0, processId, MPI_COMM_WORLD);
    }
    else
    {
        double pi = local;
        for (int id = 1; id < processCount; id++)
        {
            MPI_Recv(&local, 1, MPI_DOUBLE, id, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pi += local;
        }

        double endTime = MPI_Wtime();
        printf("pi = %f, time = %f\n", pi, endTime - startTime);
    }

    MPI_Finalize();
    return 0;
}
