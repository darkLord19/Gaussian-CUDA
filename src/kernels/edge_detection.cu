#include<time.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<cuda.h>
#include<ctime>
#include<helper_image.h>

unsigned int width, height;


int Gx[][3] = { -1 , 0 , 1 ,
				-2 , 0 , 2,
				-1 , 0 , 1	};

 int Gy[][3] = { 1, 2 ,1 ,
				 0, 0, 0,
				-1 , -2 , -1 };

int getPixel(unsigned char *org, int col, int row) {

	int sumX , sumY;
	sumX = sumY = 0;

	for (int i=-1; i<=1; i++) {
		for (int j=-1; j<=1; j++) {
			int curPixel = org [( row + j) * width + (col + i)];
			sumX += curPixel * Gx[i+1][j+1];
			sumY += curPixel * Gy[i+1][j+1];
		}
	}

	int sum = abs( sumY ) + abs( sumX );
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;
	return sum;
}

void h_EdgeDetect(unsigned char *org, unsigned char *result) {
	int offset =  1 * width;
	for (int row=1; row<height-2; row ++) {
		for (int col=1; col<width-2; col ++) {
			result [offset + col] = getPixel (org, col, row );
		}
		offset += width;
	}
}

__global__ void d_EdgeDetect(unsigned char *org, unsigned char *result, int width, int height) {
	
	int col = blockIdx .x * blockDim .x + threadIdx .x;
	int row = blockIdx .y * blockDim .y + threadIdx .y;

	if (row < 2 || col < 2 || row >= height -3 || col >= width -3 )
		return;

	int Gx[][3] = { -1 , 0 , 1 ,
					-2 , 0 , 2,
					-1 , 0 , 1	};

	 int Gy[][3] = { 1, 2 ,1 ,
					 0, 0, 0,
					-1 , -2 , -1 };

	int sumX , sumY;
	sumX = sumY = 0;

	for (int i=-1; i<= 1; i++) {
		for (int j=-1; j<=1; j++) {
			int curPixel = org [( row + j) * width + (col + i)];
			sumX += curPixel * Gx[i+1][j+1];
			sumY += curPixel * Gy[i+1][j+1];
		}
	}

	int sum = abs( sumY ) + abs( sumX );
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;

	result [row * width + col] = sum;
}

void edgeDetection(char *argv, char *h_res, char *d_res) {

/* ******************** setup work ***************************
*/

	unsigned char * d_resultPixels;
	unsigned char * h_resultPixels;
	unsigned char * h_pixels = NULL;
	unsigned char * d_pixels = NULL;

	char *srcPath = argv;
	char *h_ResultPath = h_res;
	char *d_ResultPath = d_res;

	sdkLoadPGM<unsigned char>(srcPath, &h_pixels, &width , &height);

	int ImageSize = sizeof ( unsigned char ) * width * height;

	h_resultPixels = (unsigned char *)malloc(ImageSize);
	cudaMalloc((void **) & d_pixels, ImageSize);
	cudaMalloc((void **) & d_resultPixels, ImageSize );
	cudaMemcpy(d_pixels, h_pixels, ImageSize, cudaMemcpyHostToDevice);

	 /* ******************** END setup work
	*************************** */

	 /* ************************ Host processing
	************************* */
	clock_t starttime , endtime , difference;

	starttime = clock();
	h_EdgeDetect(h_pixels , h_resultPixels);
	endtime = clock();

	difference = (endtime - starttime);

	double interval = difference / (double)CLOCKS_PER_SEC;
	printf ("CPU execution time for edge detection = %f ms\n", interval * 1000);
	sdkSavePGM<unsigned char> (h_ResultPath, h_resultPixels, width, height);
	/* ************************ END Host processing
	************************* */

	/* ************************ Device processing
	************************* */
	dim3 block(16, 16);
	dim3 grid(width / 16, height / 16);
	unsigned int timer = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* CUDA method */
	d_EdgeDetect <<< grid, block >>>(d_pixels, d_resultPixels, width, height);
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	float gpu_ms;
	cudaEventElapsedTime(&gpu_ms, start, stop);
	printf("GPU execution time for Edge Detection: %f ms\n", gpu_ms);

	cudaMemcpy(h_resultPixels, d_resultPixels, ImageSize, cudaMemcpyDeviceToHost);
	sdkSavePGM<unsigned char>(d_ResultPath, h_resultPixels, width, height);

	/* ************************ END Device processing
	************************* */
}

