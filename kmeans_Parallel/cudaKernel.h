#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void movePoints(double *devPoints, double *devSpeeds, int numOfPoints, int numDims, int numThreadsInBlock, double dt);

__global__ void computeDistancesArray(double *devPoints, double *devClusters, int numPoints, int numClusters, int numThreadsInBlock, int numDims, double *cudaDistsPointsToClusters);

__global__ void findMinDistanceFromCluster(int numPoints, int numClusters, int numThreadsInBlock, double *devDistsPointsToClusters, int *cudaPToCDictionary);

cudaError_t movePointsWithCuda(double **Points, double *devPoints,  double *devSpeeds, int numOfPoints, int numDims, double dt);

cudaError_t classifyPointsToClusters(double *devPoints, double **clusters, int numPoints, int  numClusters, int	numDims, int *pToCRelevance);
