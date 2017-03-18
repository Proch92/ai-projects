#include <stdio.h>
#include <iostream>
#include <vector>
#include <vector>
#include <future>
#include <random>
#include <stdlib.h>
#include <omp.h>
#include "math.h"

using namespace std;

//const
#define EVAPORATION_RATE 0.5 //exponential pow(pheromone, avaporation)
#define PHEROMONE_DEPOSIT 1

//definitions
class Ant {
	public:
		Ant();
		~Ant();
		void reset();
		void findpath(vector<vector<double>>);
		void walkback(vector<vector<double>>, double);
		double tourFitness();
	private:
		int pickStepLinear(int);
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
vector<vector<double>> distances;

//starts the computation
int startEuristic(int iterations, int colonySize) {
	//populating the colony
	cout << "populating the colony" << endl;
	vector<Ant> ants (colonySize);

	//creating the pheromone map
	vector<vector<double>> pheromoneMap(numberOfLocations, vector<double>(numberOfLocations, 0.0));

	//starting iterations
	for (int iter=0; iter != iterations; iter++) {
		//reinitialize ants
		for (auto &ant : ants) ant.reset();

		//ants path finding
		vector<future<void>> tasks;
		//for (auto &ant : ants) tasks.insert(async([ant, &pheromoneMap]{ant.findpath(pheromoneMap)}));
		//for (auto &task : tasks) task.get();
		for (auto &ant : ants) ant.findpath(pheromoneMap);

		//update pheromone
		double bestTour = numeric_limits<double>::max();
		for (auto &ant : ants) {
			auto tour = ant.tourFitness();
			if (tour < bestTour) bestTour = tour;
		}
		//each ant adds up his pheromone
		for (auto &ant : ants) ant.walkback(pheromoneMap, bestTour);
		
		//evaporation
		for (auto &neighbourhood : pheromoneMap)
			for (auto &edge : neighbourhood) edge = pow(edge, EVAPORATION_RATE);
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
	for (auto &p : path) p = -1;
}

void Ant::findpath(vector<vector<double>> pheromoneMap) {
	int current = 0;
	visited[0]; //first step is always 0
	int nstep = 1;

	for (auto &step : path) {
		//calc total pheromone of all paths the ant can take
		double totPheromone = 0.0;
		for (int neighbour = 0; neighbour != numberOfLocations; neighbour++) {
			if (not visited[neighbour]) totPheromone += pheromoneMap[current][neighbour];
		}
		
		//roundrobin
		//no pheromone on the road. select randomly
		//otherwise proceed with the roundrobin random selection
		if (totPheromone == 0) {
			step = pickStepLinear(nstep);
		} else {
			step = pickStepRR(pheromoneMap, totPheromone, current);
		}

		visited[step] = true;
		current = step;
		nstep++;
	}
}

void Ant::walkback(vector<vector<double>> pheromoneMap, double bestTour) {
	double quality = tourFitness() / bestTour;

	for (int i=0; i!=numberOfLocations; i++) {
		static int current = 0;

		pheromoneMap[current][path[i]] += PHEROMONE_DEPOSIT * quality;

		current = path[i];
	}
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

int Ant::pickStepLinear(int nstep) {
	int stepsRemaining = numberOfLocations - nstep;
	uniform_int_distribution<> dis(1, stepsRemaining);
	int ran = dis(randGen);

	int i;
	for (i=0; i<=ran;) {
		if (not visited[i]) i++;
	}	

	return i;
}

int Ant::pickStepRR(vector<vector<double>> pheromoneMap, double totPheromone, int current) {
	uniform_real_distribution<> dis(0, totPheromone);
	double ran = dis(randGen);

	int i;
	for (i=0; ran > 0.0; i++) {
		if (not visited[i]) ran -= pheromoneMap[current][i];
	}

	return i;
}

//main//////////////////////////////////
int main(int argc, char** argv) {
	if (argc != 4) {
		cout << "usage: tspACO iterations colony_size datafile" << endl;
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
