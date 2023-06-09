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

    // 每个进程各自找出 [3, sqrt(n)] 的素数，用作后续筛选的基数。
    LL sqrtN = (LL)sqrt(n);
    LL subLowerBound = 3;
    LL subUpperBound = sqrtN % 2 == 0 ? sqrtN - 1 : sqrtN;

    // 创建标记数组，用于标记 [3, sqrt(n)] 的每个奇数是否为合数。
    LL subSize = (subUpperBound - subLowerBound) / 2 + 1;
    bool *subCompositeFlags = (bool *)malloc(subSize);

    // 如果内存不足，就退出。
    if (subCompositeFlags == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    // 初始化标记数组全为 false。
    memset(subCompositeFlags, 0, subSize);

    // 从 3 为基数开始，检查每个奇数是否为基数的倍数。
    for (LL flagIndex = 0; flagIndex < subSize; flagIndex++)
    {
        // 如果当前基数已被标记为合数，则跳过。
        if (subCompositeFlags[flagIndex])
        {
            continue;
        }

        // 如果基数大于 sqrt(sqrt(n))，那么它的所有倍数都已经被小于基数的素数标记过了。
        LL base = flagIndex * 2 + 3;
        if (base * base > sqrtN)
        {
            break;
        }

        // 找到第一个当前基数的倍数。
        LL firstMultipleIndex = (base * base - subLowerBound) / 2;

        // 把所有当前基数的倍数都标记为合数。
        for (LL multipleIndex = firstMultipleIndex; multipleIndex < subSize; multipleIndex += base)
        {
            subCompositeFlags[multipleIndex] = true;
        }
    };

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

    // 从 3 为基数开始，检查每个奇数是否为基数的倍数。
    for (LL flagIndex = 0; flagIndex < subSize; flagIndex++)
    {
        // 如果当前基数已被标记为合数，则跳过。
        if (subCompositeFlags[flagIndex])
        {
            continue;
        }

        // 找到范围内第一个当前基数的倍数。
        LL base = flagIndex * 2 + 3;
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
    };

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
