#pragma once
#include <mpi.h>
#include "cuda_runtime.h"

void assignPointsToProc(int pointsNum, int  numOfProcs, int *sendCounts, int *displs);

double calculateDistance(int numDims, double *p1, double *p2);

double* calculateClustersDiameters(double *points, int numPoints, int numClusters, int *pToCR);

double calculateClusterQuality(double **clusters, int numClusters, double *diameters);

void kmeansCore(double **points, double *devPoints, int numPoints, int numClusters, int limit, int *pointToClusterRelevance, double **clusters, MPI_Comm   comm);

cudaError_t copyPointToGPU(double **points, double **devpoints, double **pointSpeeds, double **devSpeeds, int numpoints, int numDims);

cudaError_t FreePointDataOnGPU(double **devPoints, double **devPointSpeeds);

void InitClusterCenters(double** clusters, int k, double* points, int numOfPoints);

double* readDataFromFile(int *numPoints, int *numOfClustersToFind, double *t, double *dt, int *iterationLimit, double *qualityOfClusters, double **pointsSpeeds);

void writeAnswerToFile(double **clusters, int numClusters, double quality);