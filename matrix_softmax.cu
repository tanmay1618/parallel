#include <stdio.h>
#include "gputimer.h"

const int X= 32;		// Number of inputs
const int H= 64;		//Hidden layer size
const int K= 32;				// tile size is KxK
const int N= X*H;		//Total elements
// Utility functions: compare, print, and fill matrices
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at: %s : %d\n", file,line);
    fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);;
    exit(1);
  }
}

int compare_matrices(float *gpu, float *ref)
{
	int result = 0;

	for(int j=0; j < H; j++){
    		for(int i=0; i < X; i++)
	    		if (ref[i + j*X] != gpu[i + j*X])
	    		{
	    			// printf("reference(%d,%d) = %f but test(%d,%d) = %f\n",
	    			// i,j,ref[i+j*N],i,j,test[i+j*N]);
	    			result = 1;
	    		}
	}
    return result;
}

void print_matrix(float *mat)
{
	for(int j=0; j < N; j++) 
	{
		 printf("%4.4g ", mat[j]);
		
	}	
}
void print_debug_matrix(float *mat)
{	
	
	for(int j=0; j < N; j++) 
	{
		printf("(%4.4g,%4.4g,%4.4g)", mat[(j*3)+0 ],mat[(j*3)+1],mat[(j*3)+2]);



	}	
}
void print_input(float *mat)
{
	for(int j=0; j < X; j++) 
	{
		printf("%4.4g ", mat[j]);
		
	}	
}
// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat, int rowSize,int columnSize)
{
	for(int j=0; j < rowSize * columnSize; j++)
		mat[j] = (float) j;
}



void 
matmul_CPU(float input[],float weight[],float out[])
{
	for(int j=0; j < N; j++){
		
      			out[j] = weight[j]/2;//*input[i]; // out(j,i) = in(i,j)
		
	}
}
//matmul parallel
__global__ void 
matmul_parallel(float input[], float weight[], float out[], float d_out_debug[])
{
	//(i,j) location of element
	int intWeightX = blockIdx.x*blockDim.x + threadIdx.x;
	int intInputX = threadIdx.x;

	__shared__ float title[N];

	
	title[intWeightX]= weight[intWeightX]/2;
	d_out_debug[intWeightX*3] = intWeightX;
	d_out_debug[intWeightX*3+1] = 0;
	d_out_debug[intWeightX*3+2] = 0;
	//__syncthreads();
	// read from shared mem, coalesced write to global mem:
	out[intWeightX] = title[intWeightX];
}


int main(int argc, char **argv)
{
	
	int numbytes_input = X * sizeof(float);
	int numbytes_weight = X * H * sizeof(float);
	int numbytes_out = H * sizeof(float);

	float *input = (float *) malloc(numbytes_input);
	float *weight = (float *) malloc(numbytes_weight);
	float *out = (float *) malloc(numbytes_weight);
	float *gold = (float *) malloc(numbytes_weight);
	float *h_out_debug = (float *) malloc(numbytes_out*3);
	
	fill_matrix(input, X, 1);
	fill_matrix(weight, X, H);
	matmul_CPU(input, weight, gold);

	float *d_input, *d_weight, *d_out, *d_out_debug;

	cudaMalloc(&d_input, numbytes_input);
	cudaMalloc(&d_weight, numbytes_weight);
	cudaMalloc(&d_out, numbytes_out);
	cudaMalloc(&d_out_debug, numbytes_out*3);
	cudaMemcpy(d_input, input, numbytes_input, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, numbytes_weight, cudaMemcpyHostToDevice);

	GpuTimer timer;

/*  
 * Now time each kernel and verify that it produces the correct result.
 *
 * To be really careful about benchmarking purposes, we should run every kernel once
 * to "warm" the system and avoid any compilation or code-caching effects, then run 
 * every kernel 10 or 100 times and average the timings to smooth out any variance. 
 * But this makes for messy code and our goal is teaching, not detailed benchmarking.
 */
	timer.Start();
	dim3 blocks(N/K); // blocks per grid
	dim3 threads(K);	// threads per block
	printf("blocks %d\n",N/K);	
	print_matrix(weight);
	printf("\n");
	matmul_parallel<<<2,2>>>(d_input,d_weight, d_out,d_out_debug);
	cudaMemcpy(out, d_out, numbytes_out, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out_debug, d_out_debug, numbytes_out*3, cudaMemcpyDeviceToHost);
	printf("weight_debug\n");
	print_debug_matrix(h_out_debug);
	printf("\n");
	printf("weight\n");
		
	print_matrix(weight);
	printf("input\n");
	printf("\n");
	print_input(input);
	printf("\n");
	printf("gpu\n");
	print_matrix(out);
	printf("\n");
	printf("cpu\n");
	print_matrix(gold);
	printf("\n");
	printf("multiply_matrices: %g ms.\nVerifying transpose...%s\n",
		   timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_out);
	cudaFree(d_out_debug);
}
