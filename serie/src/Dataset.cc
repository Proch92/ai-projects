#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include "dataset.h"

using namespace std;

Dataset::Dataset(string filename) {
    x = readCSV(filename);
}

vector<float> Dataset::readCSV(string filename) {
    ifstream file(filename);

    stringstream buffer;
    buffer << file.rdbuf();
    vector<float> points;
    string point;
    while(getline(buffer, point, ',')) {
        points.push_back(stod(point));
    }

    return points;
}