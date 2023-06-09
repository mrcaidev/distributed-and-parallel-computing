#include "mpi.h"
#include <math.h>
#include <cstring>
#include <stdio.h>

#define LL long long

int main(int argc, char *argv[])
{
    // 初始化 MPI 环境。
    int processId = 0, processCount = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Barrier(MPI_COMM_WORLD);

    // 启动计时器。
    double elapsedTime = -MPI_Wtime();

    // 确保 n 已通过命令行参数传入。
    if (argc != 2)
    {
        if (processId == 0)
        {
            printf("Command line: %s <m>\n", argv[0]);
        }

        MPI_Finalize();
        exit(1);
    }

    // 从命令行参数中解析出 n。
    LL n = atoll(argv[1]);

    // 如果进程 0 不拥有 sqrt(n) 内的所有素数，就退出。
    if (2 + (n - 1) / processCount < (LL)sqrt(n))
    {
        if (processId == 0)
        {
            printf("Too many processes\n");
        }

        MPI_Finalize();
        exit(1);
    }

    // 计算当前进程的负责范围的下界。
    LL lowerBound = 2 + processId * (n - 1) / processCount;
    if (lowerBound % 2 == 0)
    {
        lowerBound++;
    }

    // 计算当前进程的负责范围的上界。
    LL upperBound = 1 + (processId + 1) * (n - 1) / processCount;
    if (upperBound % 2 == 0)
    {
        upperBound--;
    }

    // 创建标记数组，用于标记负责范围内的每个奇数是否为合数。
    LL size = (upperBound - lowerBound) / 2 + 1;
    bool *compositeFlags = (bool *)malloc(size);

    // 如果内存不足，就退出。
    if (compositeFlags == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    // 初始化标记数组全为 false。
    memset(compositeFlags, 0, size);

    // 进程 0 需要掌握它负责范围内的、尚未处理的、最小素数的索引。
    LL currentPrimeIndex = 0;

    // 从 3 为基数开始，检查每个奇数是否为基数的倍数。
    LL base = 3;
    do
    {
        // 找到范围内第一个当前基数的倍数。
        LL firstMultipleIndex = 0;
        if (base * base > lowerBound)
        {
            firstMultipleIndex = (base * base - lowerBound) / 2;
        }
        else if (lowerBound % base == 0)
        {
            firstMultipleIndex = 0;
        }
        else
        {
            firstMultipleIndex = base - lowerBound % base;
            if ((lowerBound + firstMultipleIndex) % 2 == 0)
            {
                firstMultipleIndex += base;
            }
            firstMultipleIndex /= 2;
        }

        // 把范围内所有当前基数的倍数都标记为合数。
        for (LL multipleIndex = firstMultipleIndex; multipleIndex < size; multipleIndex += base)
        {
            compositeFlags[multipleIndex] = true;
        }

        // 进程 0 更新最小素数的索引，并依此计算下一个基数。
        if (processId == 0)
        {
            while (compositeFlags[++currentPrimeIndex])
                ;
            base = 2 * currentPrimeIndex + 3;
        }

        // 其它进程从进程 0 的广播得知下一个基数。
        if (processCount > 1)
        {
            MPI_Bcast(&base, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    } while (base * base <= n);

    // 计算范围内的素数个数。
    LL count = 0;
    for (LL flagIndex = 0; flagIndex < size; flagIndex++)
    {
        if (!compositeFlags[flagIndex])
        {
            count++;
        }
    }

    // 进程间通信，计算总素数个数。
    LL globalCount = 0;
    if (processCount > 1)
    {
        MPI_Reduce(&count, &globalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // 在忽略偶数时，2 也被忽略了，所以总个数要再加 1。
    globalCount++;

    // 停止计时器。
    elapsedTime += MPI_Wtime();

    // 进程 0 输出结果。
    if (processId == 0)
    {
        printf("There are %lld primes less than or equal to %lld\n", globalCount, n);
        printf("SIEVE (%d) %10.6f\n", processCount, elapsedTime);
    }

    // 退出 MPI 环境。
    MPI_Finalize();
    return 0;
}
