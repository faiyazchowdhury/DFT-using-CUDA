#include <iostream>
#include <vector>
#include <cmath>
#include <string.h>
#include "complex.h"
#include "input_image.h"
#include <thread>

using std::vector;

struct blockData
{
    int width;
    int height;
    int expOp;
    float sumOp;
};

struct ComplexCUDA
{
    float real;
    float imag;
};
__device__ struct ComplexCUDA complexMult(struct ComplexCUDA a, struct ComplexCUDA b) {
    struct ComplexCUDA c;
    c.real = a.real*b.real-a.imag*b.imag;
    c.imag = a.real*b.imag+a.imag*b.real;
    return c;
};
__device__ struct ComplexCUDA complexAdd(struct ComplexCUDA a, struct ComplexCUDA b) {
    struct ComplexCUDA c;
    c.real = a.real+b.real;
    c.imag = a.imag+b.imag;
    return c;
};
__device__ struct ComplexCUDA createComplexCUDA() {
    struct ComplexCUDA c;
    c.real = 0;
    c.imag = 0;
    return c;
};
__device__ struct ComplexCUDA createComplexCUDA(float a) {
    struct ComplexCUDA c;
    c.real = a;
    c.imag = 0;
    return c;
};
__device__ struct ComplexCUDA createComplexCUDA(float a, float b) {
    struct ComplexCUDA c;
    c.real = a;
    c.imag = b;
    return c;
};

__global__ void blockDftHoriz(struct ComplexCUDA *dftData, struct ComplexCUDA *indata, struct blockData *bd)
{
    __shared__ struct ComplexCUDA data[2048];
    data[blockIdx.x*bd->width+threadIdx.x] = indata[blockIdx.x*bd->width+threadIdx.x];
    __shared__ struct ComplexCUDA expTerm[2048];
    __shared__ struct ComplexCUDA sum[2048];

    sum[threadIdx.x] = createComplexCUDA(0);
    if (threadIdx.x+(bd->width+1)/2 < bd->width) 
    {
        sum[threadIdx.x+(bd->width+1)/2] = createComplexCUDA(0);
    }
    __syncthreads();
    for (int t = 0; t < bd->width; t++)
    {
        expTerm[threadIdx.x] = createComplexCUDA(cos(float(bd->expOp) * 2.0 * 3.14159 * float(t) * float(threadIdx.x) / float(bd->width)),sin(bd->expOp * 2.0 * 3.14159 * float(t) * float(threadIdx.x) / float(bd->width)));
        sum[threadIdx.x] = complexAdd(sum[threadIdx.x],complexMult(data[blockIdx.x*bd->width+t],expTerm[threadIdx.x]));

        if (threadIdx.x+(bd->width+1)/2 < bd->width) 
        {
            expTerm[threadIdx.x+(bd->width+1)/2] = createComplexCUDA(cos(bd->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x+(bd->width+1)/2) / float(bd->width)),\
                    sin(bd->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x+(bd->width+1)/2) / float(bd->width)));
            sum[threadIdx.x+(bd->width+1)/2] = complexAdd(sum[threadIdx.x+(bd->width+1)/2],complexMult(data[blockIdx.x*bd->width+t],expTerm[threadIdx.x+(bd->width+1)/2]));
        }
    }
    dftData[blockIdx.x * bd->width + threadIdx.x] = complexMult(sum[threadIdx.x],createComplexCUDA(bd->sumOp));
    if (threadIdx.x+(bd->width+1)/2 < bd->width) 
    {
        dftData[blockIdx.x * bd->width + threadIdx.x+(bd->width+1)/2] = complexMult(sum[threadIdx.x+(bd->width+1)/2],createComplexCUDA(bd->sumOp));
    }
}


__global__ void blockDftVert(struct ComplexCUDA *dftData, struct ComplexCUDA *indata, struct blockData *bd)
{
    __shared__ struct ComplexCUDA data[2048];
    data[blockIdx.x*bd->width+threadIdx.x] = indata[blockIdx.x*bd->width+threadIdx.x];
    __shared__ struct ComplexCUDA expTerm[2048];
    __shared__ struct ComplexCUDA sum[2048];

    sum[threadIdx.x] = createComplexCUDA(0);
    if (threadIdx.x+(bd->width+1)/2 < bd->width) 
    {
        sum[threadIdx.x+(bd->width+1)/2] = createComplexCUDA(0);
    }
    __syncthreads();
    for (int t = 0; t < bd->height; t++)
    {
        expTerm[threadIdx.x] = createComplexCUDA(cos(bd->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x) / float(bd->height)),sin(bd->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x) / float(bd->height)));
        sum[threadIdx.x] = complexAdd(sum[threadIdx.x],complexMult(data[t*bd->width+blockIdx.x],expTerm[threadIdx.x]));

        if (threadIdx.x+(bd->height+1)/2 < bd->width) 
        {
            expTerm[threadIdx.x+(bd->height+1)/2] = createComplexCUDA(cos(bd->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x+(bd->height+1)/2) / float(bd->width)),\
                    sin(bd->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x+(bd->height+1)/2) / float(bd->width)));
            sum[threadIdx.x+(bd->height+1)/2] = complexAdd(sum[threadIdx.x+(bd->height+1)/2],complexMult(data[t*bd->width+blockIdx.x],expTerm[threadIdx.x+(bd->height+1)/2]));
        }
    }
    dftData[threadIdx.x * bd->width + blockIdx.x] = complexMult(sum[threadIdx.x],createComplexCUDA(bd->sumOp));
    if (threadIdx.x+(bd->height+1)/2 < bd->width) 
    {
        dftData[(threadIdx.x+(bd->height+1)/2) * bd->width + blockIdx.x] = complexMult(sum[threadIdx.x+(bd->height+1)/2],createComplexCUDA(bd->sumOp));
    }
}

/**
 * Do 2d dft in one thread. If forward is false, the inverse will be done
 * (forward is the default, though)
 */
Complex *doDft(Complex *data, int width, int height, bool forward = true)
{
    // std::cout << "<doDft> VAR INIT....." ;
    int expOp = forward ? -1 : 1;
    float sumOp = forward ? float(1.0) : float(1.0 / width);
    Complex *dftData2 = new Complex[width * height];
    struct ComplexCUDA dataCUDA[width*height];
    struct ComplexCUDA dftData2CUDA[width*height];
    // struct ComplexCUDA dftDataCUDA[width*height];
    // std::cout << "done!" << std::endl;    
    
    // std::cout << "<doDft> ComplexCUDA Conversion....." ;
    for (int iH=0;iH<height;iH++)
    {
        for (int iW=0;iW<width;iW++)
        {
            dataCUDA[iH*width+iW].real = data[iH*width+iW].real;
            dataCUDA[iH*width+iW].imag = data[iH*width+iW].imag;
        }
    }
    // std::cout << "done!" << std::endl;

    // std::cout << "<doDft> CUDA VAR INIT....." ;
    struct ComplexCUDA *d_data;
    struct ComplexCUDA *d_dftData;
    struct ComplexCUDA *d_dftData2;
    struct blockData *d_bd;
    // std::cout << "done!" << std::endl;

    // std::cout << "<doDft> blockData INIT....." ;
    struct blockData bd;
    bd.width = width;
    bd.height = height;
    bd.expOp = expOp;
    bd.sumOp = sumOp;
    // std::cout << "done!" << std::endl;

    // std::cout << "<doDft> CUDA MALLOC....." ;
    cudaMalloc((void **) &d_data, sizeof(dataCUDA));
    cudaMalloc((void **) &d_dftData, sizeof(dataCUDA));
    cudaMalloc((void **) &d_dftData2, sizeof(dataCUDA));
    cudaMalloc((void **) &d_bd, sizeof(bd));
    // std::cout << "done!" << std::endl;

    // std::cout << "<doDft> CUDA MEMCPY TO DEVICE....." ;
    cudaMemcpy(d_data, dataCUDA, sizeof(dataCUDA), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bd, &bd, sizeof(bd), cudaMemcpyHostToDevice);
    // std::cout << "done!" << std::endl;
    
    // std::cout << "<blockDftHoriz>....." ;
    blockDftHoriz<<<height,(width+1)/2>>>(d_dftData,d_data,d_bd);
    // blockDftHoriz<<<height,width>>>(d_dftData,d_data,d_bd);
    // std::cout << "done!" << std::endl;

    // std::cout << "<blockDftVert>....." ;
    blockDftVert<<<width,(height+1)/2>>>(d_dftData2,d_dftData,d_bd);
    // blockDftVert<<<width,height>>>(d_dftData2,d_dftData,d_bd);
    // std::cout << "done!" << std::endl;

    // std::cout << "<doDft> CUDA MEMCPY TO HOST....." ;
    // cudaMemcpy(dftData2CUDA, d_dftData, sizeof(struct ComplexCUDA[width*height]), cudaMemcpyDeviceToHost);
    cudaMemcpy(dftData2CUDA, d_dftData2, sizeof(struct ComplexCUDA[width*height]), cudaMemcpyDeviceToHost);
    // std::cout << "done!" << std::endl;

    // std::cout << "<doDft> Complex Conversion....." ;
    for (int iH=0;iH<height;iH++)
    {
        for (int iW=0;iW<width;iW++)
        {
            dftData2[iH*width+iW].real = dftData2CUDA[iH*width+iW].real;
            dftData2[iH*width+iW].imag = dftData2CUDA[iH*width+iW].imag;
        }
    }
    std::cout << "done!" << std::endl;

    // std::cout << "<doDft> CUDA MEMFREE....." ;
    cudaFree(d_data);
    cudaFree(d_dftData);
    cudaFree(d_dftData2);
    cudaFree(d_bd);
    // std::cout << "done!" << std::endl;

    return dftData2;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "wrong # inputs" << std::endl;
        return -1;
    }
    bool isForward(!strcmp(argv[1], "forward"));
    char *inputFile = argv[2];
    char *outputFile = argv[3];

    if (isForward)
    {
        std::cout << "doing forward" << std::endl;
    }
    else
    {
        std::cout << "doing reverse" << std::endl;
    }
    std::cout << inputFile << std::endl;
    InputImage im(inputFile);
    int width = im.get_width();
    int height = im.get_height();

    Complex *data = im.get_image_data();
    Complex *dftData = doDft(data, width, height, isForward);

    std::cout << "writing" << std::endl;
    im.save_image_data(outputFile, dftData, width, height);
    std::cout << "dunzo" << std::endl;
    return 0;
}
