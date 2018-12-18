#include <vector>
#include <fstream>
#include "Dataset.h"

using namespace std;

void Dataset::Dataset(string filename) {
    x = readCSV(filename);
}

vector<float> Dataset::readCSV(string filename) {
    ifstream file(filename);

    stringstream buffer;
    buffer << file.rdbuf();
    vector<float> points;
    float point;
    while(getline(buffer, point, ',')) {
        points.push_back(stod(point));
    }
}