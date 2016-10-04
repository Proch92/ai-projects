#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "math.h"

#define COLONY_SIZE 10
#define MAX_PHEROMONE 100
#define MIN_PHEROMONE 5
#define EVAPORATION 0.60

int nDataFiles = 1;
char* dataFiles[] = {"data1"};

typedef struct {
	double x, y;
} Location;

typedef struct {
	double pheromone;
	double distance;
} Edge;

struct _environment {
	int numberOfLocations;
	Location* locations;
	Edge** edges;
} environment;

int** visited; //for each ant a list of locations.
int** sortedEdges;
int** path;
double* pathFitness;

void initialize();
void sortEdgeTable();
void dealloc();
void loadData(char*);
void updatePheromone(double, double);
double bestSolution();
double worstSolution();
int pickNextStep(int, int);

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("usage: tspACO iterations\n");
		return 1;
	}

	int iterations = atoi(argv[1]);
	if (iterations < 1) {
		printf("iterations must be positive\n");
		return 1;
	}

	srand(time(NULL));
	int i, j;
	for (i=0; i!=nDataFiles; i++) {
		loadData(dataFiles[i]);
		printf("data loaded\n");
		initialize();
		printf("initialized\n");

		int iteration;
		for (iteration = 0; iteration != iterations; iteration++) {
			sortEdgeTable();
			//printf("table sorted\n");

			/*int k;
			for (j=0; j!=environment.numberOfLocations; j++) {
				for (k=0; k!=environment.numberOfLocations; k++) {
					printf("%f ", environment.edges[j][k].pheromone);
				}
				printf("\n");
			}

			printf("path choices ordered:\n");
			for (j=0; j!=environment.numberOfLocations; j++) {
				printf("from %d : ", j);
				for (k=0; k!=environment.numberOfLocations; k++) {
					printf("%d ", sortedEdges[j][k]);
				}
				printf("\n");
			}*/

			int ant;
			//#pragma omp parallel for
			for (ant=0; ant!=COLONY_SIZE; ant++) {
				int nextstep, prev = 0;
				visited[ant][0] = 0;
				path[ant][0] = 0;

				for (j=1; j!=environment.numberOfLocations; j++) {
					nextstep = pickNextStep(ant, prev);
					pathFitness[ant] += environment.edges[prev][nextstep].distance;
					prev = nextstep;
					visited[ant][nextstep] = j;
					path[ant][j] = nextstep;
				}

				nextstep = 0;
				pathFitness[ant] += environment.edges[prev][nextstep].distance;
			}

			double best = bestSolution();
			double worst = worstSolution();
			updatePheromone(best, worst);
			printf("best solution so far: %f\n", best);

			for (ant=0; ant!=COLONY_SIZE; ant++) {
				pathFitness[ant] = 0;
				for (j=0; j!=environment.numberOfLocations; j++)
					visited[ant][j] = -1;
			}
		}
	}
}

void updatePheromone(double best, double worst) {
	//evaporation
	int i, j;
	double p, trail;
	for (i=0; i!=environment.numberOfLocations; i++)
		for (j=0; j!=environment.numberOfLocations; j++)
			environment.edges[i][j].pheromone *= EVAPORATION;

	//adding trails
	for (i=0; i!=COLONY_SIZE; i++) {
		trail = 20 * (best / pathFitness[i]) / COLONY_SIZE;
		
		int prev = 0;
		for (j=1; j!=environment.numberOfLocations; j++) {
			p = environment.edges[path[i][prev]][path[i][j]].pheromone;
			p += trail;
			if (p > MAX_PHEROMONE)
				p = MAX_PHEROMONE;
			if (p < MIN_PHEROMONE)
				p = MIN_PHEROMONE;

			environment.edges[path[i][prev]][path[i][j]].pheromone = p;
			environment.edges[path[i][j]][path[i][prev]].pheromone = p;

			prev = j;
		}

		p = environment.edges[path[i][prev]][0].pheromone;
		p += trail;
		if (p > MAX_PHEROMONE)
			p = MAX_PHEROMONE;
		if (p < MIN_PHEROMONE)
			p = MIN_PHEROMONE;

		environment.edges[path[i][prev]][0].pheromone = p;
		environment.edges[0][path[i][prev]].pheromone = p;
	}
}

//roulette random pick
int pickNextStep(int ant, int prev) {
	double tot = 0;

	int i;
	for (i=0; i!=environment.numberOfLocations; i++)
		if (visited[ant][i] == -1)
			tot += environment.edges[prev][i].pheromone;

	double r;
	if (tot < 1)
		r = 0;
	else
		r = rand() % (int)tot;
	double accumulator = 0;
	int a;
	for (i=0; i!=environment.numberOfLocations; i++) {
		a = sortedEdges[prev][i];
		if (visited[ant][a] == -1) {
			accumulator += environment.edges[prev][a].pheromone;
			if (accumulator >= r)
				return a;
		}
	}

	return 0;
}

double bestSolution() {
	int i;
	double best = pathFitness[0];
	for (i=1; i!=COLONY_SIZE; i++)
		if (pathFitness[i] < best)
			best = pathFitness[i];

	return best;
}

double worstSolution() {
	int i;
	double worst = 0;
	for (i=0; i!=COLONY_SIZE; i++)
		if (pathFitness[i] > worst)
			worst = pathFitness[i];

	return worst;
}

static int cmpFitness(const void* e1, const void* e2, void* i) {
	return (environment.edges[*(int*)i][*(int*)e2].pheromone)-(environment.edges[*(int*)i][*(int*)e1].pheromone);
}

void sortEdgeTable() {
	int i;
	for (i=0; i!=environment.numberOfLocations; i++)
		qsort_r(sortedEdges[i], environment.numberOfLocations, sizeof(int), cmpFitness, &i);
}

void initialize() {
	environment.edges = (Edge**) malloc(environment.numberOfLocations * sizeof(Edge*));
	int i, j;
	for (i=0; i!=environment.numberOfLocations; i++)
		environment.edges[i] = (Edge*) malloc(environment.numberOfLocations * sizeof(Edge));

	sortedEdges = (int**) malloc(environment.numberOfLocations * sizeof(int*));
	for (i=0; i!=environment.numberOfLocations; i++)
		sortedEdges[i] = (int*) malloc(environment.numberOfLocations * sizeof(int));

	Location* loc = environment.locations;
	for (i=0; i!=environment.numberOfLocations; i++) {
		for (j=0; j!=environment.numberOfLocations; j++) {
			environment.edges[i][j].pheromone = MIN_PHEROMONE;
			environment.edges[i][j].distance = (sqrtf(powf(loc[i].x - loc[j].x, 2) + powf(loc[i].y - loc[j].y, 2)));

			sortedEdges[i][j] = j;
		}
		environment.edges[i][i].pheromone = 0;
	}

	visited = (int**) malloc(sizeof(int*) * COLONY_SIZE);
	for (i=0; i!=COLONY_SIZE; i++) {
		visited[i] = (int*) malloc(sizeof(int) * environment.numberOfLocations);
		for (j=0; j!=environment.numberOfLocations; j++) {
			visited[i][j] = -1;
		}
	}

	path = (int**) malloc(sizeof(int*) * COLONY_SIZE);
	for (i=0; i!=COLONY_SIZE; i++) {
		path[i] = (int*) malloc(sizeof(int) * environment.numberOfLocations);
		for (j=0; j!=environment.numberOfLocations; j++) {
			path[i][j] = -1;
		}
	}

	pathFitness = (double*) malloc(sizeof(double) * COLONY_SIZE);
}

void dealloc() {
	int i, j;
	for (i=0; i!=environment.numberOfLocations; i++) {
		free(environment.edges[i]);
		free(sortedEdges);
	}
	free(environment.edges);
	free(sortedEdges);

	for (i=0; i!=environment.numberOfLocations; i++)
		free(visited[i]);
	free(visited);

	for (i=0; i!=environment.numberOfLocations; i++)
		free(path[i]);
	free(path);

	free(pathFitness);
}

void loadData(char* path) {
   FILE* datafile = fopen(path, "r");
   int nloc;
   fread(&nloc, sizeof(int), 1, datafile);
	printf("nloc = %d\n", nloc);
   environment.numberOfLocations = nloc;
   environment.locations = (Location*) malloc(sizeof(Location) * nloc);
   fread(environment.locations, sizeof(double)*2, nloc, datafile);
   fclose(datafile);
}

