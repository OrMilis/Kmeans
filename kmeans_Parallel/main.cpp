#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>
#include "Header.h"
#include "cudaKernel.h"

#define NUM_OF_INTS 4
#define NUM_OF_DOUBLE 2
#define INPUT_FILE "D:\\input.txt"
#define OUTPUT_FILE "D:\\output.txt"
#define MASTERID 0
#define DIMS 2

int* intMemoryAllocation(int memorySize);

int main(int argc, char *argv[])
{

	int numprocs, myid;

	int i, j, k, totalPoints, limit, *pointToClusterDictionary, pointsInProc, *pToCDictionaryEachProc, *sendNum = NULL, *recvNum = NULL,			
		*recvNumUpdatedPoints, *displsScatter = NULL, *spaceInArr = NULL, *spaceInArrAfterUpdate, *intFileData;							

	double	 *pointsArr, *cudaPoints = NULL, *cudaVelocity = NULL, *diameters, **clusters, requiredQuality, currentQuality, **pointsProc,
		*pointsVelocityArr, **pointsVelocityProc, *doubleFileData, t, dt, currentTime = 0;				

	//Init MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

	intFileData = (int*)malloc(NUM_OF_INTS * sizeof(int));
	doubleFileData = (double*)malloc(NUM_OF_DOUBLE * sizeof(double));

	sendNum = intMemoryAllocation(numprocs);
	displsScatter = intMemoryAllocation(numprocs);
	recvNum = intMemoryAllocation(numprocs);
	spaceInArr = intMemoryAllocation(numprocs);
	recvNumUpdatedPoints = intMemoryAllocation(numprocs);
	spaceInArrAfterUpdate = intMemoryAllocation(numprocs);

	double startTime = MPI_Wtime();

	if (myid == MASTERID)
	{
		
		pointsArr = readDataFromFile(&totalPoints, &k, &t, &dt, &limit, &requiredQuality, &pointsVelocityArr);

		pointToClusterDictionary = intMemoryAllocation(totalPoints);

		intFileData[0] = totalPoints;
		intFileData[1] = limit;
		intFileData[2] = k;
		intFileData[3] = t;

		doubleFileData[0] = requiredQuality;
		doubleFileData[1] = dt;
	}

	MPI_Bcast(intFileData, NUM_OF_INTS, MPI_INT, MASTERID, MPI_COMM_WORLD);
	MPI_Bcast(doubleFileData, NUM_OF_DOUBLE, MPI_DOUBLE, MASTERID, MPI_COMM_WORLD);
	
	totalPoints = intFileData[0];
	limit = intFileData[1];
	k = intFileData[2];
	t = intFileData[3];

	requiredQuality = doubleFileData[0];
	dt = doubleFileData[1];

	assignPointsToProc(totalPoints, numprocs, sendNum, displsScatter);

	for (i = 0; i < numprocs; ++i) 
	{
		//Put the number recived into recvArr, num of points in procs
		recvNum[i] = sendNum[i] / DIMS; 
	}
	spaceInArr[0] = 0;
	for (i = 1; i < numprocs; ++i) 
	{
		//Put space between point in array
		spaceInArr[i] = spaceInArr[i - 1] + recvNum[i - 1]; 
	}

	for (i = 0; i < numprocs; ++i) 
	{
		//num of points updated
		recvNumUpdatedPoints[i] = recvNum[i] * DIMS; 
	}
	spaceInArrAfterUpdate[0] = 0;
	for (i = 1; i < numprocs; ++i) 
	{ 
		//Put number of updated points
		spaceInArrAfterUpdate[i] = spaceInArrAfterUpdate[i - 1] + recvNumUpdatedPoints[i - 1]; 
	}

	pointsInProc = sendNum[myid] / DIMS;

	pointsProc = (double**)malloc(pointsInProc * sizeof(double*));
	pointsProc[0] = (double*)malloc(pointsInProc * DIMS * sizeof(double));
	for (i = 1; i < pointsInProc; ++i)
	{
		pointsProc[i] = pointsProc[i - 1] + DIMS;
	}
	pointsVelocityProc = (double**)malloc(pointsInProc * sizeof(double*));
	pointsVelocityProc[0] = (double*)malloc(pointsInProc * DIMS * sizeof(double));
	for (i = 1; i < pointsInProc; ++i)
	{
		pointsVelocityProc[i] = pointsVelocityProc[i - 1] + DIMS;
	}

	MPI_Scatterv(pointsArr, sendNum, displsScatter, MPI_DOUBLE, pointsProc[0], sendNum[myid], MPI_DOUBLE, MASTERID, MPI_COMM_WORLD);
	MPI_Scatterv(pointsVelocityArr, sendNum, displsScatter, MPI_DOUBLE, pointsVelocityProc[0], sendNum[myid], MPI_DOUBLE, MASTERID, MPI_COMM_WORLD);
	
	copyPointToGPU(pointsProc, &cudaPoints, pointsVelocityProc, &cudaVelocity, pointsInProc, DIMS);
	pToCDictionaryEachProc = intMemoryAllocation(pointsInProc);
	clusters = (double**)malloc(k * sizeof(double*));
	clusters[0] = (double*)malloc(k * DIMS * sizeof(double));
	for (i = 1; i < k; ++i)
	{
		clusters[i] = clusters[i - 1] + DIMS;
	}

	if (myid == MASTERID)
		InitClusterCenters(clusters, k, pointsArr, totalPoints);
	MPI_Bcast(clusters[0], k * DIMS, MPI_DOUBLE, MASTERID, MPI_COMM_WORLD);
	
	do
	{
		if (currentTime != 0)
		{
			movePointsWithCuda(pointsProc, cudaPoints, cudaVelocity, pointsInProc, DIMS, dt);;
		}

		kmeansCore(pointsProc, cudaPoints, pointsInProc, k, limit, pToCDictionaryEachProc, clusters, MPI_COMM_WORLD);
		
		MPI_Gatherv(pToCDictionaryEachProc, pointsInProc, MPI_INT, pointToClusterDictionary, recvNum, spaceInArr, MPI_INT, MASTERID, MPI_COMM_WORLD);
		MPI_Gatherv(pointsProc[0], pointsInProc, MPI_DOUBLE, pointsArr, recvNumUpdatedPoints, spaceInArrAfterUpdate, MPI_DOUBLE, MASTERID, MPI_COMM_WORLD);

		if (myid == MASTERID)
		{
			diameters = calculateClustersDiameters(pointsArr, totalPoints, k, pointToClusterDictionary);

			currentQuality = calculateClusterQuality(clusters, k, diameters);

			free(diameters);
		}

		MPI_Bcast(&currentQuality, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (myid == MASTERID)
		{
			printf("dT = %0.2f ---> QualityM = %lf\n", currentTime, currentQuality);
			fflush(stdout);
		}

		currentTime += dt;
			
	} while (currentTime < t && currentQuality > requiredQuality);

	if (myid == MASTERID)
		writeAnswerToFile(clusters, k, currentQuality);
	
	//Time measurment
	double endTime = MPI_Wtime() - startTime;

	//free all memory allocated
	FreePointDataOnGPU(&cudaPoints, &cudaVelocity);
	free(clusters[0]);
	free(recvNumUpdatedPoints);
	free(spaceInArrAfterUpdate);
	free(doubleFileData);
	free(clusters);
	free(pToCDictionaryEachProc);
	free(spaceInArr);
	free(intFileData);
	free(pointsProc[0]);
	free(pointsProc);
	free(sendNum);
	free(displsScatter);
	free(recvNum);


	if (myid == MASTERID)
	{
		free(pointsArr);
		free(pointsVelocityArr);
		free(pointToClusterDictionary);
		printf("\nTime worked = %.5f\nQuality Mesure = %.5f\n\n", endTime, currentQuality);
	}

	MPI_Finalize();
}

void assignPointsToProc(int pointsNum,int  numOfProcs,int *sendCounts, int *displs)
{
	int i, remainder, index, *pointCounterForProc;

	pointCounterForProc = intMemoryAllocation(numOfProcs);

	remainder = pointsNum % numOfProcs;
	index = 0;
#pragma omp parallel for
	for (i = 0; i < numOfProcs; ++i)
	{
		pointCounterForProc[i] = pointsNum / numOfProcs;
		if (remainder > 0)
		{
			pointCounterForProc[i]++;
			remainder--;
		}

		sendCounts[i] = pointCounterForProc[i] * DIMS;
		displs[i] = index;
		index += sendCounts[i];
	}
	free(pointCounterForProc);
}

int* intMemoryAllocation(int memorySize) {
	return (int*)malloc(memorySize * sizeof(int));
}

//Quality checks
double calculateDistance(int numDims, double *p1, double *p2)
{
	int i;
	double dist = 0.0;

	for (i = 0; i < numDims; ++i)
	{
		dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
	}

	return sqrt(dist);
}

double* calculateClustersDiameters(double *points, int numPoints, int numClusters, int *pointToClusterDictionary) // Calculates Diameters
{
	double diameter, dist, *diametersThreads, *diameters;
	int i, j, threadsNum, tid, stride;

	diameter = 0.0;

	threadsNum = omp_get_max_threads();

	diametersThreads = (double*)calloc(threadsNum  * numClusters, sizeof(double));

	diameters = (double*)malloc(numClusters * sizeof(double));

#pragma omp parallel for private(j, tid, dist, stride) shared(diametersThreads) //Calculate Distance with OMP
	for (i = 0; i < numPoints; ++i)
	{
		tid = omp_get_thread_num();
		stride = tid * numClusters;

		for (j = i + 1; j < numPoints; ++j)
		{
			if (pointToClusterDictionary[i] == pointToClusterDictionary[j])
			{
				dist = calculateDistance(DIMS, points + (i * DIMS), points + (j * DIMS));
				if (dist > diametersThreads[stride + pointToClusterDictionary[i]])
					diametersThreads[stride + pointToClusterDictionary[i]] = dist;
			}
		}

	}
#pragma omp parallel for // each thread handle a cluster (if clusternum <= threadnum)
	for (i = 0; i < numClusters; i++)
	{
		diameters[i] = diametersThreads[i];
		for (j = 1; j < threadsNum; j++)
		{
			if (diameters[i] < diametersThreads[j * numClusters + i])
				diameters[i] = diametersThreads[j * numClusters + i];
		}
	}

	free(diametersThreads);

	return diameters;
}

double calculateClusterQuality(double **clusters, int numClusters, double *diameters)
{
	int i, j;

	int numElements = numClusters * (numClusters - 1);
	double quality = 0.0;

#pragma omp parallel for private(j) reduction(+ : quality) // Calculate all qualitys and Sum with omp
	for (i = 0; i < numClusters; ++i)
	{
		for (j = i + 1; j < numClusters; ++j)
		{
			quality += (diameters[i] + diameters[j]) / calculateDistance(DIMS, clusters[i], clusters[j]);
		}
	}
	return quality / numElements;
}
//Quality checks

//Kmeans
void kmeansCore(double **points, double *devPoints, int numPoints, int numClusters, int limit, int *pointToClusterDictionary, double **clusters, MPI_Comm   comm)
{
	int i, j, index, loop = 0, sumDelta = 0, delta;
	int *cudaPToCDictionary;
	int *newClusterSize;
	int *clusterSize;  
	double **newClusters;

#pragma omp parallel for // Init the Dictionary with OMP
	for (i = 0; i < numPoints; ++i)
	{
		pointToClusterDictionary[i] = -1;
	}

	cudaPToCDictionary = (int*)malloc(numPoints * sizeof(int));

	newClusterSize = (int*)calloc(numClusters, sizeof(int));

	clusterSize = (int*)calloc(numClusters, sizeof(int));

	newClusters = (double**)malloc(numClusters * sizeof(double*));

	newClusters[0] = (double*)calloc(numClusters * DIMS, sizeof(double));

	for (i = 1; i < numClusters; ++i)
	{
		newClusters[i] = newClusters[i - 1] + DIMS;
	}

	do
	{
		delta = 0;

		classifyPointsToClusters(devPoints, clusters, numPoints, numClusters, DIMS, cudaPToCDictionary);

		for (i = 0; i < numPoints; ++i)
		{
			if (pointToClusterDictionary[i] != cudaPToCDictionary[i])
			{
				delta++;
				pointToClusterDictionary[i] = cudaPToCDictionary[i];
			}

			index = cudaPToCDictionary[i];

			newClusterSize[index]++;

			for (j = 0; j < DIMS; ++j)
				newClusters[index][j] += points[i][j];
		}

		MPI_Allreduce(&delta, &sumDelta, 1, MPI_INT, MPI_SUM, comm);

		if (sumDelta == 0)
			break;

		MPI_Allreduce(newClusters[0], clusters[0], numClusters * DIMS, MPI_DOUBLE, MPI_SUM, comm);

		MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT, MPI_SUM, comm);

		for (i = 0; i < numClusters; i++)
		{
			for (j = 0; j < DIMS; j++)
			{
				if (clusterSize[i] > 1)
				{
					clusters[i][j] /= clusterSize[i];
				}
				newClusters[i][j] = 0.0;
			}
			newClusterSize[i] = 0;
		}

	} while (++loop < limit);

	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
	free(clusterSize);
	free(cudaPToCDictionary);

}

cudaError_t copyPointToGPU(double **points, double **devpoints, double **pointSpeeds, double **devSpeeds, int numpoints, int numDims)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)devpoints, numpoints * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(devpoints);
	}

	cudaStatus = cudaMalloc((void**)devSpeeds, numpoints * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(devpoints);
	}

	cudaStatus = cudaMemcpy(*devpoints, points[0], numpoints * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(devpoints);
	}

	cudaStatus = cudaMemcpy(*devSpeeds, pointSpeeds[0], numpoints * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(devpoints);
	}

	return cudaStatus;
}

cudaError_t FreePointDataOnGPU(double **devPoints, double **devPointSpeeds)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaFree(*devPoints);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed!");
	}

	cudaStatus = cudaFree(*devPointSpeeds);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed!");
	}

	return cudaStatus;
}

void InitClusterCenters(double** clusters, int k, double* points, int numOfPoints)
{
	int i, j;

	for (i = 0; i < k; ++i)
	{
		for (j = 0; j < DIMS; ++j)
		{
			clusters[i][j] = points[j + i * DIMS];
		}
	}
}
//Kmeans

//IO
double* readDataFromFile(int *numPoints, int	*numOfClustersToFind, double *t, double *dt, int *iterationLimit, double *qualityOfClusters, double **pointsSpeeds) {
	int i, j, counter = 0;
	double *points;
	FILE *f;

	f = fopen(INPUT_FILE, "r");

	fscanf(f, "%d %d %lf %lf %d %lf\n", numPoints, numOfClustersToFind, t, dt, iterationLimit, qualityOfClusters);

	points = (double*)malloc((*numPoints) * DIMS * sizeof(double));
	*pointsSpeeds = (double*)malloc((*numPoints) * DIMS * sizeof(double));

	for (i = 0; i < (*numPoints); ++i)
	{
		for (j = 0; j < DIMS; ++j)
		{
			fscanf(f, "%lf ", &points[j + i* DIMS]);
		}
		for (j = 0; j < DIMS; ++j)
		{
			fscanf(f, "%lf ", (*pointsSpeeds) + j + i* DIMS);
		}

		fscanf(f, "\n");
	}

	fclose(f);
	return points;
}

void writeAnswerToFile(double **clusters, int numClusters, double quality)		//quality of the cluster group found
{
	int i, j;

	FILE *f = fopen(OUTPUT_FILE, "w");

	fprintf(f, "Quality Mesure: %1f\n\n", quality);
	fprintf(f, "K = %d\n", numClusters);
	fprintf(f, "Cluster Centers:\n\n");

	for (i = 0; i < numClusters; ++i)
	{
		fprintf(f, "Cluster %d: ", i + 1);

		for (j = 0; j < DIMS; ++j)
		{
			fprintf(f, "[%.2f] ", clusters[i][j]);
		}

		fprintf(f, "\n\n");
	}
	fclose(f);
}
//IO