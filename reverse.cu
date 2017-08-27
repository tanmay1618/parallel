#include <stdio.h>
#include "gputimer.h"

const int N = 64;

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat, int rowSize,int columnSize)
{
	for(int j=0; j < rowSize * columnSize; j++)
		mat[j] = (float) j;
}

void print_Matrix(float *mat, int numRows, int numColumns){
	for(int i = 0; i < numRows; i++){
		for(int j = 0; j<numColumns; j++){
			printf("%4.4g ", mat[j + i*numColumns]);
		}
	printf("\n");
	}
			
}
void
reverse_cpu(float* input, float* output){
	for(int i =0; i<N; i++){
		output[N-1-i]=input[i];
	}
}

__global__ void
reverse_parallel(float* input, float* output){
	int myIdx = threadIdx.x;
	output[blockDim.x - 1 - myIdx] = input[myIdx];
	//output[myIdx] = blockDim.x - 1 - myIdx;
}

int main(int argc, char **argv)
{
	int numbytes = N*sizeof(float);	
	float* input = (float*) malloc(numbytes);
	float* output = (float*) malloc(numbytes);
	float* gold = (float*) malloc(numbytes);
	printf("%d",numbytes);
	fill_matrix(input,1,N);
	reverse_cpu(input,gold);
	
	float *d_in, *d_out;
	
	cudaMalloc(&d_in,numbytes);
	cudaMalloc(&d_out,numbytes);
	cudaMemcpy(d_in,input,numbytes,cudaMemcpyHostToDevice);

	GpuTimer timer;
	printf("Input\n");
	print_Matrix(input,1,N);
	timer.Start();
	dim3 blocks(1); // blocks per grid
	dim3 threads(N);	// threads per block
	reverse_parallel<<<blocks,threads>>>(d_in,d_out);
	cudaMemcpy(output,d_out,numbytes,cudaMemcpyDeviceToHost);
	printf("Output\n");
	print_Matrix(output,1,N);


}
