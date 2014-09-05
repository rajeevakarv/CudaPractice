/****

Author: Rajeev Verma
Contact : rv4560
This program should print the CUDA capabilities available in your system. 

****/


#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include <cuda_runtime_api.h>
#include "common.h"
// Entry point for the program.
bool checkForError(cudaError_t Error){

	if (Error==cudaSuccess)
		return 1;
	else{
		printf("Cuda API failed with error: %s", cudaGetErrorString);
		return 0;
	}

}

int main()
{
//Keep the error status
cudaError_t status;

/*cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaDriverGetVersion(int* driverVersion);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
*/

printf("CUDA Device Capabilities:\n\n");
int count;
int driverVersion;
int runtimeVersion;
int device;
int i;



status = cudaGetDeviceCount(&count);
if(!checkForError(status))
	return 0;
status= cudaDriverGetVersion(&driverVersion);
if(!checkForError(status))
	return 0;

status = cudaRuntimeGetVersion(&runtimeVersion);
if(!checkForError(status))
	return 0;

printf("CUDA Devices Found: %d", count);
printf("\nCUDA Driver: %d", driverVersion);
printf("\nCUDA Runtime: %d\n\n", runtimeVersion);


for(i=0; i<count; i++){
	cudaDeviceProp prop;
	status = cudaGetDeviceProperties(&prop, i);
	if(!checkForError(status))
		return 0;
	if(prop.integrated)
		printf("Device %d: %s (Integrated)\n",i, prop.name, prop.integrated);
	else
		printf("Device %d: %s (Discrete)\n",i, prop.name, prop.integrated);
	printf("CUDA Capability %d.%d\n", prop.major, prop.minor);
	printf("Processing: \n");
	printf("\tMultiprocessors : %d\n", prop.multiProcessorCount);
	printf("\tMax Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("\tMax Block Size: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("\tThreads per block : %d\n", prop.maxThreadsPerBlock);
	printf("\tThreads per Multiprocessor : %d\n", prop.maxThreadsPerMultiProcessor);
	printf("\tWarp Size: %d\n", prop.warpSize);
	printf("\tClock Rate: %fGHz\n", (float)prop.clockRate/1000000);
	printf("Memory:\n");
	printf("\tGlobal: %d MB\n", prop.totalGlobalMem>>20);
	printf("\tConstant: %d KB\n", prop.totalConstMem>>10);
	printf("\tShared/blk: %d KB\n", prop.sharedMemPerBlock>>10);
	printf("\tRegisters/blk: %d\n", prop.regsPerBlock);
	printf("\tMaximun Pitch: %d MB\n", prop.memPitch>>20);
	printf("\tTexture Alignment: %d B\n", prop.textureAlignment);
	printf("\tL2 Cache Size: %d B\n", prop.l2CacheSize);
	printf("\tClock Rate: %dMHz\n", prop.memoryClockRate/1000);
	if (prop.deviceOverlap)
		printf("Concurrent copy & Execute: Yes\n");
	else
		printf("Concurrent copy & Execute: No\n");
	if(prop.kernelExecTimeoutEnabled)
		printf("Kernel Time Limit: Yes\n");
	else
		printf("Kernel Time Limit: No\n");
	if (prop.canMapHostMemory)
		printf("Supports Page-Locked Memory Mapping: Yes\n");
	else
		printf("Supports Page-Locked Memory Mapping: No\n");
	if (!prop.computeMode)
		printf("Compute Mode: Default\n\n");
	else
		printf("Compute Mode: Not the Default\n\n");
}

getchar();
return 0;
}

