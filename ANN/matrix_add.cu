#include <stdio.h>
#include "gputimer.h"
//size of matrix is MxN
const int M = 1024; //number of rows
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
exp_CPU(float* input1, float* input2, int numRows, int numColumns, float* out){
	for(int i = 0; i< numRows; i++){
		for(int j = 0; j< numColumns; j++){
			out[i] = input2[i*numColumns + j]  + input1[i*numColumns + j] ;
			/*if(i==1369){
				printf("%4.4g %4.4g ,",weight[i*numColumns + j],input[j]);
			}*/
		}
	}
}
__global__ void
parallel_sum(float* in1,float* in2, float* out){
	
	//int numTot = gridDim.y*blockDim.y;
	int myIdx = blockDim.x*blockIdx.x + threadIdx.x;
	
	out[myIdx] = in1[myIdx]+in2[myIdx];

}

int main(int argc, char **argv)
{
	int numbytes_input = M * sizeof(float);
	int numbytes_output = M *sizeof(float);
	
	float* input2 = (float*) malloc(numbytes_input);
	float* input1 = (float*) malloc(numbytes_input);
	float* out = (float*) malloc(numbytes_output);
	float* gold = (float*) malloc(numbytes_output);
	
	fill_matrix(input1,M,1);
	fill_matrix(input2,M,1);
	exp_CPU(input1, input2 ,M,1,gold);	
	//print_Matrix(gold,4,1);
	//print_Matrix_row(gold,1369,1);
	//printf("input^\n");
	//print_Matrix(input2,4,1);
	float *d_in1,*d_in2,*d_out;
	
	cudaMalloc(&d_in1,numbytes_input);
	cudaMalloc(&d_in2,numbytes_input);
	cudaMalloc(&d_out,numbytes_output);
	cudaMemcpy(d_in1,input1,numbytes_input,cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2,input2,numbytes_input,cudaMemcpyHostToDevice);
	
	GpuTimer timer;
	//print_Matrix(gold,N,1);
	//printf("\n");
	//matrix mul
	dim3 blocks(M/32);
	dim3 threads(32);
	timer.Start();
	parallel_sum<<<blocks,threads>>>(d_in1,d_in2,d_out);
	cudaMemcpy(out,d_out,numbytes_output,cudaMemcpyDeviceToHost);

	printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", 
	       timer.Elapsed(), compare_matrices(out, gold,M,1) ? "Failed" : "Success");		
	//print_Matrix(out,4,1);
	//print_Matrix_row(out,1369,1);
	
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	
	
}
