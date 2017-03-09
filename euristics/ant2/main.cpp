#include <stdio.h>
#include <iostream>
#include <vector>
#include <array>
#include <async>
#include <future>
#include <stdlib.h>
#include <omp.h>
#include "math.h"

using namespace std;

//const
#define EVAPORATION_RATE 0.5

//definitions
class Ant {
	public:
		Ant();
		~Ant();
		void reset();
		void findpath(vector<vector<double> >);
		void walkback(vector<vector<double> >);
	private:
		vector<bool> visited;
		vector<int> path;
};

typedef struct {
	double x, y;
} Location;

//globals
int numberOfLocations;
Location* locations;

//starts the computation
int startEuristic(int iterations, int colonySize) {
	//populating the colony
	cout << "populating the colony" << endl;
	vector<Ant> ants (colonySize);

	//creating the pheromone map
	array<array<double> > pheromoneMap(numberOfLocations, vector<double>(numberOfLocations, 0.0));

	//starting iterations
	for (int iter=0; iter != iterations; iter++) {
		//reinitialize ants
		for (auto ant : ants) ant.reset();

		//ants path finding
		vector<future<void> > tasks;
		for (auto ant : ants) tasks.insert(async(&Ant::findpath, &ant, pheromoneMap));
		for (auto task : tasks) task.get();

		//update pheromone
		//each ant adds up his pheromone
		for (auto ant : ants) ant.walkback(pheromoneMap);
		
		//evaporation
		for (auto neighborhood : pheromoneMap)
			for (auto &edge : neighborhood) edge = edge * (1 - EVAPORATION_RATE);
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
}

//ANT/////////////////////
Ant::Ant() {
	visited.resize(numberOfLocations);
	path.resize(numberOfLocations);
	reset();
}

Ant::~Ant() {}

void Ant::reset() {
	for (auto &v : visited) v = false;
	for (auto &p : path) p = -1;
}

void Ant::findpath(array<array<double> > pheromoneMap) {
}

void Ant::walkback(array<array<double> > pheromoneMap) {
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
