#include "mpi.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int processId = 0, processCount = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Barrier(MPI_COMM_WORLD);

    double elapsedTime = -MPI_Wtime();

    if (argc != 2)
    {
        if (processId == 0)
        {
            printf("Command line: %s <m>\n", argv[0]);
        }

        MPI_Finalize();
        exit(1);
    }

    int n = atoi(argv[1]);

    int lowerBound = 2 + processId * (n - 1) / processCount;
    int upperBound = 1 + (processId + 1) * (n - 1) / processCount;
    int size = upperBound - lowerBound + 1;

    if (2 + size < (int)sqrt((double)n))
    {
        if (processId == 0)
        {
            printf("Too many processes\n");
        }

        MPI_Finalize();
        exit(1);
    }

    char *compositeFlags = (char *)malloc(size);

    if (compositeFlags == NULL)
    {
        if (processId == 0)
        {
            printf("Cannot allocate enough memory\n");
        }

        MPI_Finalize();
        exit(1);
    }

    for (int i = 0; i < size; i++)
    {
        compositeFlags[i] = 0;
    }

    int currentPrimeIndex;
    if (processId == 0)
    {
        currentPrimeIndex = 0;
    }

    int base = 2;
    do
    {
        int firstCompositeIndex = 0;

        if (base * base > lowerBound)
        {
            firstCompositeIndex = base * base - lowerBound;
        }
        else if (lowerBound % base == 0)
        {
            firstCompositeIndex = 0;
        }
        else
        {
            firstCompositeIndex = base - lowerBound % base;
        }

        for (int i = firstCompositeIndex; i < size; i += base)
        {
            compositeFlags[i] = 1;
        }

        if (processId == 0)
        {
            while (compositeFlags[++currentPrimeIndex])
                ;
            base = currentPrimeIndex + 2;
        }

        if (processCount > 1)
        {
            MPI_Bcast(&base, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    } while (base * base <= n);

    int count = 0;
    for (int i = 0; i < size; i++)
    {
        if (compositeFlags[i] == 0)
        {
            count++;
        }
    }

    int globalCount = 0;
    if (processCount > 1)
    {
        MPI_Reduce(&count, &globalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    elapsedTime += MPI_Wtime();

    if (processId == 0)
    {
        printf("There are %d primes less than or equal to %d\n", globalCount, n);
        printf("SIEVE (%d) %10.6f\n", processCount, elapsedTime);
    }

    MPI_Finalize();
    return 0;
}
