#include "mpi.h"
#include <math.h>
#include <stdio.h>

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
    int n = atoi(argv[1]);

    // 如果进程数过多，就退出。
    if (2 + (n - 1) / processCount < (int)sqrt((double)n))
    {
        if (processId == 0)
        {
            printf("Too many processes\n");
        }

        MPI_Finalize();
        exit(1);
    }

    // 每个进程各自找出 [3, sqrt(n)] 的素数，用作后续筛选的基数。
    int sqrtN = (int)sqrt((double)n);
    int subLowerBound = 3;
    int subUpperBound = sqrtN % 2 == 0 ? sqrtN - 1 : sqrtN;

    // 创建标记数组，用于标记 [3, sqrt(n)] 的每个奇数是否为合数。
    int subSize = (subUpperBound - subLowerBound) / 2 + 1;
    bool *subCompositeFlags = (bool *)malloc(subSize);

    // 如果内存不足，就退出。
    if (subCompositeFlags == NULL)
    {
        if (processId == 0)
        {
            printf("Cannot allocate enough memory\n");
        }

        MPI_Finalize();
        exit(1);
    }

    // 初始化标记数组全为 false。
    for (int flagIndex = 0; flagIndex < subSize; flagIndex++)
    {
        subCompositeFlags[flagIndex] = false;
    }

    // 从 3 为基数开始，检查每个奇数是否为基数的倍数。
    for (int flagIndex = 0; flagIndex < subSize; flagIndex++)
    {
        // 如果当前基数已被标记为合数，则跳过。
        if (subCompositeFlags[flagIndex] == true)
        {
            continue;
        }

        // 找到第一个当前基数的倍数。
        int base = flagIndex * 2 + 3;
        int firstMultipleIndex = (base * base - subLowerBound) / 2;

        // 把所有当前基数的倍数都标记为合数。
        for (int multipleIndex = firstMultipleIndex; multipleIndex < subSize; multipleIndex += base)
        {
            subCompositeFlags[multipleIndex] = true;
        }
    };

    // 计算当前进程的负责范围的下界。
    int lowerBound = 2 + processId * (n - 1) / processCount;
    if (lowerBound % 2 == 0)
    {
        lowerBound++;
    }

    // 计算当前进程的负责范围的上界。
    int upperBound = 1 + (processId + 1) * (n - 1) / processCount;
    if (upperBound % 2 == 0)
    {
        upperBound--;
    }

    // 负责范围内的数字总数。
    int totalSize = (upperBound - lowerBound) / 2 + 1;

    // 根据缓存大小，将负责范围划分为多个块。
    int CACHE_SIZE = 4194304;
    int blockSize = CACHE_SIZE / sizeof(int) / processCount;
    int blockNum = totalSize / blockSize;
    if (totalSize % blockSize != 0)
    {
        blockNum++;
    }

    // 创建标记数组，用于标记块内的每个奇数是否为合数。
    bool *compositeFlags = (bool *)malloc(blockSize);

    // 如果内存不足，就退出。
    if (compositeFlags == NULL)
    {
        if (processId == 0)
        {
            printf("Cannot allocate enough memory\n");
        }

        MPI_Finalize();
        exit(1);
    }

    // 创建计数器，用于记录负责范围内的素数个数。
    int count = 0;

    // 依次处理每个块。
    for (int blockIndex = 0; blockIndex < blockNum; blockIndex++)
    {
        // 重置标记数组全为 false。
        for (int flagIndex = 0; flagIndex < blockSize; flagIndex++)
        {
            compositeFlags[flagIndex] = false;
        }

        // 计算当前块的下界和上界。
        int blockLowerBound = lowerBound + blockIndex * blockSize * 2;
        int blockUpperBound = blockLowerBound + blockSize * 2 - 1;
        if (blockUpperBound > upperBound)
        {
            blockUpperBound = upperBound;
        }
        if (blockUpperBound % 2 == 0)
        {
            blockUpperBound--;
        }
        int realBlockSize = (blockUpperBound - blockLowerBound) / 2 + 1;

        // 从 3 为基数开始，检查每个奇数是否为基数的倍数。
        for (int flagIndex = 0; flagIndex < subSize; flagIndex++)
        {
            // 如果当前基数已被标记为合数，则跳过。
            if (subCompositeFlags[flagIndex])
            {
                continue;
            }

            // 找到范围内第一个当前基数的倍数。
            int base = flagIndex * 2 + 3;
            int firstMultipleIndex = 0;
            if (base * base > blockLowerBound)
            {
                firstMultipleIndex = (base * base - blockLowerBound) / 2;
            }
            else if (blockLowerBound % base == 0)
            {
                firstMultipleIndex = 0;
            }
            else
            {
                firstMultipleIndex = base - blockLowerBound % base;
                if ((blockLowerBound + firstMultipleIndex) % 2 == 0)
                {
                    firstMultipleIndex += base;
                }
                firstMultipleIndex /= 2;
            }

            // 把范围内所有当前基数的倍数都标记为合数。
            for (int multipleIndex = firstMultipleIndex; multipleIndex < realBlockSize; multipleIndex += base)
            {
                compositeFlags[multipleIndex] = true;
            }
        };

        // 计算范围内的素数个数。
        for (int flagIndex = 0; flagIndex < realBlockSize; flagIndex++)
        {
            if (!compositeFlags[flagIndex])
            {
                count++;
            }
        }
    }

    // 进程间通信，计算总素数个数。
    int globalCount = 0;
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
        printf("There are %d primes less than or equal to %d\n", globalCount, n);
        printf("SIEVE (%d) %10.6f\n", processCount, elapsedTime);
    }

    // 退出 MPI 环境。
    MPI_Finalize();
    return 0;
}
