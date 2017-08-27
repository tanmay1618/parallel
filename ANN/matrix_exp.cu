#include <stdio.h>
#include "gputimer.h"
//size of matrix is MxN
const int N = 64; //number of rows
const int M = 32; //number of columns
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
int compare_matrices(float *gpu, float *ref)
{
	int result = 0;

	for(int j=0; j < M; j++)
    	for(int i=0; i < N; i++)
    		if (ref[i + j*N] != gpu[i + j*N])
    		{
    			// printf("reference(%d,%d) = %f but test(%d,%d) = %f\n",
    			// i,j,ref[i+j*N],i,j,test[i+j*N]);
    			result = 1;
    		}
    return result;
}
void
exp_CPU(float* in, int numRows, int numColumns, float* out){
	for(int i = 0; i< numRows; i++){
		for(int j = 0; j< numColumns; j++){
			out[i*numColumns + j] = in[i*numColumns + j]/2;
		}
	}
}
__global__ void
parallel_exp(float* in, float* out){
	int myIdx = blockDim.x*blockIdx.x + threadIdx.x;
	out[myIdx] = in[myIdx]/2;

}
int main(int argc, char **argv)
{
	int numbytes = N * M * sizeof(float);
	float* in = (float*) malloc(numbytes);
	float* out = (float*) malloc(numbytes);
	float* gold = (float*) malloc(numbytes);
	
	fill_matrix(in,N,M);
	exp_CPU(in,N,M,gold);	
	
	//print_Matrix(in,N,M);
	
	float *d_in,*d_out;
	
	cudaMalloc(&d_in,numbytes);
	cudaMalloc(&d_out,numbytes);
	cudaMemcpy(d_in,in,numbytes,cudaMemcpyHostToDevice);
	
	GpuTimer timer;
	
	dim3 blocks(64);
	dim3 threads(32);
	timer.Start();
	parallel_exp<<<blocks,threads>>>(d_in,d_out);
	cudaMemcpy(out,d_out,numbytes,cudaMemcpyDeviceToHost);
	printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", 
	       timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");		
	//print_Matrix(gold,N,M);
	

	
	
}
