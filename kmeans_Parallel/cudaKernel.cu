
#include "cudaKernel.h"

__global__ void movePoints(double *points, double *velocity, int numOfPoints, int numDims, int numThreadsInBlock, double dt)
{
	int blockID = blockIdx.x;
	int numOfCoord = numOfPoints * numDims;
	int i;

	//because we optimized num of blocks before calling the calling
	if ((blockID == gridDim.x - 1) && (numOfPoints % blockDim.x <= threadIdx.x)) { return; } //dismiss spare threads

	for (i = 0; i < numDims; ++i)
	{
		points[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] += velocity[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] * dt;		
	}
}

__global__ void computeDistancesArray(double *devPoints, double *devClusters, int numPoints, int numClusters, int numThreadsInBlock, int numDims, double *cudaDistsPointsToClusters)
{
	int i;
	int blockID = blockIdx.x;
	double result = 0;

	if ((blockID == gridDim.x - 1) && (numPoints % blockDim.x <= threadIdx.x)) { return; } //dismiss spare threads

	//each thread computes a distance in a matrix of distances
	for (i = 0; i < numDims; ++i)
	{
		result += (devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] - devClusters[threadIdx.y * numDims + i]) * (devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] - devClusters[threadIdx.y * numDims + i]);
	}

	//this array contains for each point its distance from each cluster
	cudaDistsPointsToClusters[numPoints*threadIdx.y + (blockID * numThreadsInBlock + threadIdx.x)] = result;
}

__global__ void findMinDistanceFromCluster(int numPoints, int numClusters, int numThreadsInBlock, double *devDistsPointsToClusters, int *cudaPToCDictionary) // calculate min distance from cluster for each point
{
	int i;
	int xid = threadIdx.x;
	int blockId = blockIdx.x;
	double minIndex = 0;
	double minDistance, tempDistance;

	if ((blockIdx.x == gridDim.x - 1) && (numPoints % blockDim.x <= xid)) { return; }  

	minDistance = devDistsPointsToClusters[(numThreadsInBlock * blockId) + xid];

	for (i = 1; i < numClusters; ++i)
	{
		tempDistance = devDistsPointsToClusters[(numThreadsInBlock * blockId) + xid + (i*numPoints)];
		if (minDistance > tempDistance)
		{
			minIndex = i;
			minDistance = tempDistance;
		}
	}

	cudaPToCDictionary[numThreadsInBlock * blockId + xid] = minIndex;
}

cudaError_t movePointsWithCuda(double **points, double *devPoints, double *devSpeeds, int numOfPoints, int numDims, double dt)
{
	cudaError_t cudaStatus;
	cudaDeviceProp cudaProp; //used to retrieve specs from GPU

	int numBlocks, numThreadsInBlock;

	cudaGetDeviceProperties(&cudaProp, 0); // 0 is for device 0

	numThreadsInBlock = cudaProp.maxThreadsPerBlock;
	numBlocks = numOfPoints / numThreadsInBlock;
	
	if (numOfPoints % numThreadsInBlock > 0) { numBlocks++; }

	movePoints<<<numBlocks, numThreadsInBlock>>>(devPoints, devSpeeds, numOfPoints, numDims, numThreadsInBlock, dt);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	//update the points from gpu to cpu
	cudaStatus = cudaMemcpy((void**)points[0], devPoints, numOfPoints * numDims * sizeof(double), cudaMemcpyDeviceToHost);

Error:
	return cudaStatus;
}

cudaError_t classifyPointsToClusters(double *devPoints, double **clusters, int numPoints, int numClusters, int numDims, int *pToCRelevance)
{
	cudaError_t cudaStatus;
	cudaDeviceProp cudaProp; //used to retrieve specs from GPU

	int maxThreadsPerBlock;
	int numBlocks, numThreadsInBlock;

	cudaGetDeviceProperties(&cudaProp, 0); // 0 is for device 0

	//configuring kernel params
	numThreadsInBlock = cudaProp.maxThreadsPerBlock / numClusters;
	dim3 dim(numThreadsInBlock, numClusters);
	numBlocks = numPoints / numThreadsInBlock;

	if (numPoints % numThreadsInBlock > 0) { numBlocks++; }

	double *devClusters;
	double *devDistsPointsToClusters = 0;
	int   *devPToCRelevance = 0;

	// Allocate GPU buffers for three points (two input, one output) 
	cudaStatus = cudaMalloc((void**)&devClusters, numClusters * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devDistsPointsToClusters, numClusters * numPoints * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devPToCRelevance, numPoints * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(devClusters, clusters[0], numClusters * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//launch kernels//
	computeDistancesArray <<<numBlocks, dim >>> (devPoints, devClusters, numPoints, numClusters, numThreadsInBlock, numDims, devDistsPointsToClusters);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//reconfiguring params for next kernel
	numThreadsInBlock = cudaProp.maxThreadsPerBlock;
	numBlocks = numPoints / numThreadsInBlock;
	if (numPoints % numThreadsInBlock > 0) { numBlocks++; }

	findMinDistanceFromCluster <<<numBlocks, numThreadsInBlock >>> (numPoints, numClusters, numThreadsInBlock, devDistsPointsToClusters, devPToCRelevance);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(pToCRelevance, devPToCRelevance, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(devClusters);
	cudaFree(devDistsPointsToClusters);
	cudaFree(devPToCRelevance);

	return cudaStatus;
}
