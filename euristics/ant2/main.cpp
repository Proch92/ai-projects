#include <stdio.h>
#include <iostream>
#include <vector>
#include <future>
#include <thread>
#include <random>
#include <stdlib.h>
#include <omp.h>
#include "math.h"

using namespace std;

//const
#define EVAPORATION_RATE 0.3
#define MIN_PHEROMONE 0.01
#define MAX_PHEROMONE 1000
#define PHEROMONE_DEPOSIT 10

//definitions
class Ant {
	public:
		Ant();
		~Ant();
		void reset();
		void findpath(vector<vector<double>>);
		void walkback(vector<vector<double>>&, double, double);
		double tourFitness();
	private:
		int pickStepRR(vector<vector<double>>, double, int);

		vector<bool> visited;
		vector<int> path;
		random_device randGen;
};

typedef struct {
	double x, y;
} Location;

//globals
int numberOfLocations;
Location* locations;
double** distances;

//starts the computation
int startEuristic(int iterations, int colonySize) {
	//populating the colony
	cout << "populating the colony" << endl;
	vector<Ant> ants (colonySize);

	//creating the pheromone map
	cout << "creating pheromone map" << endl;
	vector<vector<double>> pheromoneMap(numberOfLocations, vector<double>(numberOfLocations, MIN_PHEROMONE));

	//starting iterations
	cout << "starting iterations" << endl;
	for (int iter=0; iter < iterations; iter++) {
		cout << "iter: " << iter << endl;
		//reinitialize ants
		for (auto &ant : ants) ant.reset();

		//ants path finding
		vector<thread> tasks;
		for (auto &ant : ants) tasks.push_back(thread(&Ant::findpath, &ant, pheromoneMap));
		for (auto &task : tasks) task.join();

		//update pheromone
		//evaporation
		for (auto &neighbourhood : pheromoneMap) {
			for (auto &edge : neighbourhood) {
				edge *= EVAPORATION_RATE;
				if (edge < MIN_PHEROMONE) edge = MIN_PHEROMONE;
			}
		}

		//best and worst tours
		double bestTour = numeric_limits<double>::max();
		for (auto &ant : ants) {
			auto tour = ant.tourFitness();
			if (tour < bestTour) bestTour = tour;
		}
		double worstTour = 0;
		for (auto &ant : ants) {
			auto tour = ant.tourFitness();
			if (tour > worstTour) worstTour = tour;
		}
		cout << "best tour fitness " << bestTour << endl;
		cout << "worst tour fitness " << worstTour << endl;

		//each ant adds up his pheromone
		for (auto &ant : ants) ant.walkback(pheromoneMap, bestTour, worstTour);
	}
	
	return 0;
}

void loadData(const char* path) {
	FILE* datafile = fopen(path, "r");
	
	fread(&numberOfLocations, sizeof(int), 1, datafile);
	cout << "# of locations: " << numberOfLocations << endl;
	locations = (Location*) malloc(sizeof(Location) * numberOfLocations);
	fread(locations, sizeof(Location), numberOfLocations, datafile);
	fclose(datafile);


	distances = (double**) malloc(sizeof(double*) * numberOfLocations);
	for (int i=0; i!=numberOfLocations; i++) {
		distances[i] = (double*) malloc(sizeof(double) * numberOfLocations);
	}

	for (int i=0; i!=numberOfLocations; i++)	{
		for (int j=0; j!=numberOfLocations; j++) {
			distances[i][j] = sqrt(pow(locations[i].x - locations[j].x, 2) + pow(locations[i].y - locations[j].y, 2));
		}
	}
}

//ANT/////////////////////
Ant::Ant() {
	visited.resize(numberOfLocations);
	path.resize(numberOfLocations - 1); //first step is always 0
	reset();

	random_device randGen;
}

Ant::~Ant() {}

void Ant::reset() {
	for (auto &&v : visited) v = false;
}

void Ant::findpath(vector<vector<double>> pheromoneMap) {
	int current = 0;
	visited[0] = true; //first step is always 0
	int nstep = 1;

	for (auto &step : path) {
		//calc total pheromone of all paths the ant can take
		double totPheromone = 0.0;
		for (int neighbour = 0; neighbour != numberOfLocations; neighbour++) {
			if (not visited[neighbour]) totPheromone += pheromoneMap[current][neighbour];
		}
		
		//roundrobin
		step = pickStepRR(pheromoneMap, totPheromone, current);

		visited[step] = true;
		current = step;
		nstep++;
	}
}

void Ant::walkback(vector<vector<double>> &pheromoneMap, double bestTour, double worstTour) {
	double fitness = tourFitness();
	double quality = (worstTour - fitness) / (worstTour - bestTour);
	double deposit = PHEROMONE_DEPOSIT * quality;

	int current = 0;
	for (int i=0; i!=numberOfLocations; i++) {
		pheromoneMap[current][path[i]] += deposit;
		if (pheromoneMap[current][path[i]] > MAX_PHEROMONE) pheromoneMap[current][path[i]] = MAX_PHEROMONE;
		current = path[i];
	}

	pheromoneMap[current][0] += deposit;
	if (pheromoneMap[current][0] > MAX_PHEROMONE) pheromoneMap[current][0] = MAX_PHEROMONE;
}

double Ant::tourFitness() {
	double fitness = 0;
	for (auto step : path) {
		static int current = 0;

		fitness += distances[current][step];

		current = step;
	}

	return fitness;
}

int Ant::pickStepRR(vector<vector<double>> pheromoneMap, double totPheromone, int current) {
	uniform_real_distribution<> dis(0, totPheromone);
	double ran = dis(randGen);

	int i;
	for (i=0; ran > 0.0; i++) {
		if (not visited[i]) ran -= pheromoneMap[current][i];
	}

	return i-1;
}

//main//////////////////////////////////
int main(int argc, char** argv) {
	if (argc != 4) {
		cout << "usage: " << argv[0] << " iterations colony_size datafile" << endl;
		exit(1);
	}

	int iterations = atoi(argv[1]);
	if (iterations < 1) {
		cout << "iterations must be positive" << endl;
		exit(2);
	}

	int colonySize = atoi(argv[2]);
	if (colonySize < 1) {
		cout << "colony size must be positive" << endl;
		exit(3);
	}

	// load data
	loadData(argv[3]);

	if (startEuristic(iterations, colonySize) == 0) {
		cout << "Run successful" << endl;
	}
	else {
		cout << "Run exited with problems" << endl;
	}

	return 0;
}
