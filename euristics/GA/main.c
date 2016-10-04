#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <omp.h>

#define POP_SIZE 10000
#define P_CROSSOVER 60
#define P_MUTATION 1

int nDataFiles = 1;
char* datafiles[] = {"data1"};

typedef struct {
	double x, y;
} Location;

typedef struct {
	int* path;
	double fitness;
} Genome;

typedef struct {
	int size;
	double totFitness;
	Genome* agents;
} Population;

struct _environment {
	int numberOfLocations;
	Location* locations;
	Population population;
} environment;

void loadData(char* path);
void initialize();
void dealloc();
double fitness(Genome);
int pickParent();
void sortGenomesByFitness();
void qsortGenomesByFitness();
void crossover(Genome, Genome, Genome);
void bestParent(Genome, Genome, Genome);
void mutate(Genome);
double distance(int, int);

Population nextGeneration;

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("usage: tspGA iteration_number");
		return 1;
	}
	int iterations = atoi(argv[1]);
	if (iterations < 1) {
		printf("iterations must a positive number");
		return 1;
	}
	
	srand(time(NULL));

	int i;
	for (i=0; i!=nDataFiles; i++) {
		loadData(datafiles[i]);
		initialize();

		int iteration;
		for (iteration = 0; iteration != iterations; iteration++) {
			printf("iteration %d ----------------------------------------------------\n", iteration);
			qsortGenomesByFitness();	
			printf("best fitness so far: %f\n", environment.population.agents[0].fitness);
			//riproduzione
			int i;
			#pragma omp parallel for
			for (i=0; i!=environment.population.size; i++)
			{
				int parent1, parent2;
				parent1 = pickParent();
				parent2 = pickParent();

				if ((rand() % 100) < P_CROSSOVER) {
					crossover(environment.population.agents[parent1], environment.population.agents[parent2], nextGeneration.agents[i]);
				}
				else {
					bestParent(environment.population.agents[parent1], environment.population.agents[parent2], nextGeneration.agents[i]);
				}

				if ((rand() % 100) < P_MUTATION)
					mutate(nextGeneration.agents[i]);
			}

			//kill old generation and calculate new fitness
			//swapping new generation over the old one
			Genome* tmp = environment.population.agents;
			environment.population.agents = nextGeneration.agents;
			nextGeneration.agents = tmp;
			#pragma omp parallel for
			for (i=0; i!=environment.population.size; i++)
			{
				//calculate firness for the new generation
				environment.population.agents[i].fitness = fitness(environment.population.agents[i]);
			}
			
			environment.population.totFitness = 0;
			for (i=0; i!=environment.population.size; i++)
				environment.population.totFitness += environment.population.agents[i].fitness;
		}

		dealloc();
	}
}

double fitness(Genome agent) {
	double tot = 0;
	
	int i;
	for(i=0; i!=environment.numberOfLocations; i++) {
		tot += distance(agent.path[i], agent.path[(i+1)%environment.numberOfLocations]);
	}

	return tot;
} 

void mutate(Genome agent) {
	int p1 = rand() % environment.numberOfLocations;
	int p2 = rand() % environment.numberOfLocations;

	int tmp = agent.path[p1];
	agent.path[p1] = agent.path[p2];
	agent.path[p2] = tmp;
}

void bestParent(Genome parent1, Genome parent2, Genome child) {
	Genome best = (parent1.fitness < parent2.fitness) ? parent1 : parent2;

	int i;
	for(i=0; i!=environment.numberOfLocations; i++) {
		child.path[i] = best.path[i];
	}
}

// double pivot crossover
void crossover(Genome parent1, Genome parent2, Genome child) {
	int pivot1 = rand() % environment.numberOfLocations;
	int pivot2 = rand() % environment.numberOfLocations;

	if (pivot2 < pivot1) {
		int tmp = pivot2;
		pivot2 = pivot1;
		pivot1 = tmp;
	}

	int i;
	for (i=0; i!=environment.numberOfLocations; i++) {
		if (i >= pivot1 && i <= pivot2)
			child.path[i] = parent2.path[i];
		else
			child.path[i] = parent1.path[i];
	}
}

static int cmpFitness(const void *a1, const void *a2) {
	return ((Genome*)a1)->fitness - ((Genome*)a2)->fitness;
}

void qsortGenomesByFitness() {
	qsort(environment.population.agents, environment.population.size, sizeof(Genome), cmpFitness);
}

void sortGenomesByFitness() {
	char sorted = 0;

	int i;
	while(!sorted) {
		sorted = 1;

		//#pragma omp parallel for
		for (i=0; i<=POP_SIZE-1; i=i+2)
		{
			if (environment.population.agents[i].fitness > environment.population.agents[i+1].fitness) {
				sorted = 0;

				int tmpi = environment.population.agents[i].fitness;
				environment.population.agents[i].fitness = environment.population.agents[i+1].fitness;
				environment.population.agents[i+1].fitness = tmpi;
				
				int* tmppath = environment.population.agents[i].path;
				environment.population.agents[i].path = environment.population.agents[i+1].path;
				environment.population.agents[i+1].path = tmppath;
			}
		}

		//#pragma omp parallel for
		for (i=1; i<=POP_SIZE-1; i=i+2)
		{
			if (environment.population.agents[i].fitness > environment.population.agents[i+1].fitness) {
				sorted = 0;

				int tmpi = environment.population.agents[i].fitness;
				environment.population.agents[i].fitness = environment.population.agents[i+1].fitness;
				environment.population.agents[i+1].fitness = tmpi;
				
				int* tmppath = environment.population.agents[i].path;
				environment.population.agents[i].path = environment.population.agents[i+1].path;
				environment.population.agents[i+1].path = tmppath;
			}
		}
	}
}

int pickParent() {
	double totFitness = rand() % (int)(environment.population.totFitness);

	int i;
	for (i=0; i!=environment.population.size; i++) {
		totFitness -= environment.population.agents[i].fitness;
		if (totFitness <= 0)
			return i;
	}
}

void initialize() {
	environment.population.size = POP_SIZE;
	environment.population.agents = (Genome*) malloc(sizeof(Genome) * POP_SIZE);	
	if (environment.population.agents == NULL) {
		puts("error allocating memory for agents");
		exit(-1);
	}

	int i, k;
	for (i=0; i != environment.population.size; i++) {
		environment.population.agents[i].path = (int*) malloc(sizeof(int) * environment.numberOfLocations);
		if(environment.population.agents[i].path == NULL) {
			printf("error allocating memory for path. i:%d\n", i);
			exit(-1);
		}

		for (k=0; k != environment.numberOfLocations; k++)
			environment.population.agents[i].path[k] = k;

		//randomize genomes
		for (k=environment.numberOfLocations-1; k >= 0; k--) {
			int r = rand() % (k+1);

			int temp = environment.population.agents[i].path[k];
			environment.population.agents[i].path[k] = environment.population.agents[i].path[r];
			environment.population.agents[i].path[r] = temp;
		}
		environment.population.agents[i].fitness = fitness(environment.population.agents[i]);
	}
	
	nextGeneration.size = environment.population.size;
	nextGeneration.agents = (Genome*) malloc(sizeof(Genome) * nextGeneration.size);
	for (i=0; i!=POP_SIZE; i++)
		nextGeneration.agents[i].path = (int*) malloc(sizeof(int) * environment.numberOfLocations);

	environment.population.totFitness = 0;
	for (i=0; i!=environment.population.size; i++)
		environment.population.totFitness += environment.population.agents[i].fitness;
}

void loadData(char* path) {
	FILE* datafile = fopen(path, "r");
	int nloc;
	fread(&nloc, sizeof(int), 1, datafile);
	environment.numberOfLocations = nloc;
	environment.locations = (Location*) malloc(sizeof(Location) * nloc);
	fread(environment.locations, sizeof(double)*2, nloc, datafile);
	fclose(datafile);
}

void dealloc() {
	int i;
	for(i=0; i != POP_SIZE; i++) {
		free(environment.population.agents[i].path);
		free(nextGeneration.agents[i].path);
	}
	free(environment.population.agents);
	free(nextGeneration.agents);
	free(environment.locations);
}

double distance(int l1, int l2) {
	Location* loc = environment.locations;
	return (sqrtf(powf(loc[l2].x - loc[l1].x, 2) + powf(loc[l2].y - loc[l1].y, 2)));	
}
