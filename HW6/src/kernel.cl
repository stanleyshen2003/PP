__kernel void convolution(
    __global float* inputImage,
    __global float* filter,
    __global float* outputImage,
    int filterWidth)
{
    int i = get_global_id(0); // Row index
    int j = get_global_id(1); // Column index
    int imageHeight = get_global_size(0);
    int imageWidth = get_global_size(1);
    int halffilterSize = filterWidth / 2;
    float sum = 0;

    int row_start, row_end, col_start, col_end;
    row_start = - max(0, (i - halffilterSize));
    col_start = - max(0, (j - halffilterSize));
    row_end =  max(0, i + halffilterSize - imageWidth);
    col_end = max(0, j + halffilterSize - imageHeight);

    // from serial implemetnation
    for (int k = -halffilterSize + row_start; k <= halffilterSize - row_end; k++) {
        for (int l = -halffilterSize + col_start; l <= halffilterSize - col_end; l++) {
            int row = i + k;
            int col = j + l;

            // Check boundaries
            if (row >= 0 && row < imageHeight && col >= 0 && col < imageWidth) {
                sum += inputImage[row * imageWidth + col] * filter[(k + halffilterSize) * filterWidth + (l + halffilterSize)];
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}