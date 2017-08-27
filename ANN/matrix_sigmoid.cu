#include <stdio.h>
#include "gputimer.h"
//size of matrix is MxN
const int N = 128; //number of rows
//const int M = 64; //number of columns
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
void print_Matrix_row(float *mat, int numRows, int numColumns){
	
		for(int j = 0; j<numColumns; j++){
			printf("%4.4g ", mat[j + numRows*numColumns]);
		}
	printf("\n");
	
			
}
int compare_matrices(float *gpu, float *ref, int intRows, int intCols)
{
	int result = 0;

	for(int j=0; j < intRows; j++)
    	for(int i=0; i < intCols; i++)
    		if (ref[i + j*intCols] != gpu[i + j*intCols])
    		{
    			// printf("reference(%d,%d) = %f but test(%d,%d) = %f\n",
    			// i,j,ref[i+j*N],i,j,test[i+j*N]);
    			result = 1;
    		}
    return result;
}
void
exp_CPU(float* input, int numRows, int numColumns, float* out){

	for(int i = 0; i< numRows; i++){
		for(int j = 0; j< numColumns; j++){
			out[i*numColumns + j] = input[i*numColumns + j]*2 ;
		}
	}
}
__global__ void
parallel_softmax(float* in, float* out){


	int myIdx = blockDim.x*blockIdx.x + threadIdx.x;

	out[myIdx] = in[myIdx]*2;
	
}

int main(int argc, char **argv)
{
	int numbytes_input = N * sizeof(float);
	int numbytes_output =  N *sizeof(float);
	
	float* input = (float*) malloc(numbytes_input);
	float* out = (float*) malloc(numbytes_output);
	float* gold = (float*) malloc(numbytes_output);
	

	fill_matrix(input,N,1);
	exp_CPU(input,N,1,gold);	
	//print_Matrix(input,4,1);
	//print_Matrix_row(gold,1369,1);
	//printf("input^\n");
	float *d_in,*d_out;
	
	cudaMalloc(&d_in,numbytes_input);
	cudaMalloc(&d_out,numbytes_output);
	cudaMemcpy(d_in,input,numbytes_input,cudaMemcpyHostToDevice);
	
	GpuTimer timer;
	print_Matrix(gold,4,1);
	//printf("\n");
	//matrix mul
	dim3 blocks(1);
	dim3 threads(N);
	timer.Start();
	parallel_softmax<<<blocks,threads>>>(d_in,d_out);
	cudaMemcpy(out,d_out,numbytes_output,cudaMemcpyDeviceToHost);

	printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", 
	       timer.Elapsed(), compare_matrices(out, gold,N,1) ? "Failed" : "Success");		
	print_Matrix(out,4,1);
	//print_Matrix_row(out,1369,1);
	
	cudaFree(d_in);	
	cudaFree(d_out);

	
	
}
