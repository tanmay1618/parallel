#include <stdio.h>
#include "gputimer.h"
//size of matrix is MxN
const int N = 128; //number of rows
const int M = 64; //number of columns
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
exp_CPU(float* input, float* weight, int numRows, int numColumns, float* out){
	for(int i = 0; i< numRows; i++){
		for(int j = 0; j< numColumns; j++){
			out[i] += weight[i*numColumns + j] * input[j] ;
			/*if(i==1369){
				printf("%4.4g %4.4g ,",weight[i*numColumns + j],input[j]);
			}*/
		}
	}
}
__global__ void
parallel_exp(float* w,float* in, float* out){
	
	int numTot = gridDim.y*blockDim.y;
	int myIdx = blockDim.x*blockIdx.x + threadIdx.x;
	int myIdy = blockDim.y*blockIdx.y + threadIdx.y;
	
	out[myIdx*numTot+myIdy] = in[myIdy]*w[myIdx*numTot+myIdy];//w[myIdx*numTot+myIdy];
	//out[myIdx] = temp[myIdx*numTot];
	//int myIdx = blockDim.x*blockIdx.x + threadIdx.x;
	//out[myIdx] = w[myIdx]*in[threadIdx.x]; for block = ro and
	//out[myIdx] = w[myIdx]*in[threadIdx.x%4];4 i column ie 

}
__global__ void
parallel_exp_reduce_sum(float* in,float* out){
	int numTot = blockDim.x;
	int myIdx = blockDim.x*blockIdx.x + threadIdx.x;
	int myIdy = threadIdx.x;
	__shared__ float temp[64];
	
	temp[myIdy] = in[myIdx];
	__syncthreads();

	for (unsigned int s = numTot / 2; s > 0; s >>= 1)
    	{
        	if (myIdy < s)
        	{	
			temp[myIdy] += temp[myIdy + s];
        	}
       		 __syncthreads();        // make sure all adds at one stage are done!
	}
	if(myIdy==0){
		out[blockIdx.x] = temp[myIdy];
	}

}
int main(int argc, char **argv)
{
	int numbytes_weight = N * M * sizeof(float);
	int numbytes_input = M * sizeof(float);
	int numbytes_output_temp = M * N * sizeof(float);
	int numbytes_output =  N *sizeof(float);
	
	float* weight = (float*) malloc(numbytes_weight);
	float* input = (float*) malloc(numbytes_input);
	float* out_temp = (float*) malloc(numbytes_output_temp);
	float* out = (float*) malloc(numbytes_output);
	float* gold = (float*) malloc(numbytes_output);
	
	fill_matrix(weight,N,M);
	fill_matrix(input,M,1);
	exp_CPU(input, weight ,N,M,gold);	
	//print_Matrix(gold,N,1);
	//print_Matrix_row(gold,1369,1);
	//printf("input^\n");
	float *d_w,*d_in,*d_out, *d_out_temp;
	
	cudaMalloc(&d_in,numbytes_input);
	cudaMalloc(&d_out,numbytes_output);
	cudaMalloc(&d_out_temp,numbytes_output_temp);
	cudaMalloc(&d_w,numbytes_weight);
	cudaMemcpy(d_in,input,numbytes_input,cudaMemcpyHostToDevice);
	cudaMemcpy(d_w,weight,numbytes_weight,cudaMemcpyHostToDevice);
	
	GpuTimer timer;
	//print_Matrix(gold,N,1);
	//printf("\n");
	//matrix mul
	dim3 blocks(4,2);
	dim3 threads(32,32);
	timer.Start();
	parallel_exp<<<blocks,threads>>>(d_w,d_in,d_out_temp);
	//cudaMemcpy(out_temp,d_out_temp,numbytes_output_temp,cudaMemcpyDeviceToHost);
	//print_Matrix(out_temp,4,M);
	//reduce
	//cudaMemcpy(d_w,out_temp,numbytes_input,cudaMemcpyHostToDevice);
	dim3 blocks1(N);
	dim3 threads1(M);
	parallel_exp_reduce_sum<<<blocks1,threads1>>>(d_out_temp,d_out);
	cudaMemcpy(out,d_out,numbytes_output,cudaMemcpyDeviceToHost);

	printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", 
	       timer.Elapsed(), compare_matrices(out, gold,N,1) ? "Failed" : "Success");		
	//print_Matrix(out,N,1);
	//print_Matrix_row(out,1369,1);
	
	cudaFree(d_in);
	cudaFree(d_w);
	cudaFree(d_out);
	cudaFree(d_out_temp);
	
	
}
