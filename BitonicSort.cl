__kernel void BitonicSort(
    __global int* data,
    const uint stage,
    const uint passOfStage,
    const uint totalSize)
{
    uint idx = get_global_id(0);
    uint pairDistance = 1 << (stage - passOfStage);
    uint blockWidth   = 2 * pairDistance;
    uint leftId       = (idx / pairDistance) * blockWidth + (idx % pairDistance);
    uint rightId      = leftId + pairDistance;

    if (rightId < totalSize) {
        int leftVal  = data[leftId];
        int rightVal = data[rightId];

        // Направление сортировки зависит от индекса первого элемента в паре (leftId)
        bool ascending = ((leftId / (1 << stage)) % 2) == 0;

        if ((leftVal > rightVal) == ascending) {
            data[leftId]  = rightVal;
            data[rightId] = leftVal;
        }
    }
}