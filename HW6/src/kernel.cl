__kernel void convolution(
    __global float* inputImage,
    __global float* filter,
    __global float* outputImage,
    int filterWidth)
{
    int i = get_global_id(0); // Row index
    int j = get_global_id(1); // Column index
    int imageHeight = get_global_size(1);
    int imageWidth = get_global_size(0);
    int halffilterSize = filterWidth / 2;
    float sum = 0;

    int row_start, row_end, col_start, col_end;
    row_start = max(0, (halffilterSize - j));
    col_start = max(0, (halffilterSize - i));
    row_end =  max(0, j + halffilterSize - imageHeight - 1);
    col_end = max(0, i + halffilterSize - imageWidth - 1);

    // from serial implemetnation
    for (int k = -halffilterSize + row_start; k <= halffilterSize - row_end; k++) {
        for (int l = -halffilterSize + col_start; l <= halffilterSize - col_end; l++) {
            int row = j + k;
            int col = i + l;

            // Check boundaries
            sum += inputImage[row * imageWidth + col] * filter[(k + halffilterSize) * filterWidth + (l + halffilterSize)];
        }
    }
    outputImage[j * imageWidth + i] = sum;
}