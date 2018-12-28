#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

#include "dataset.h"

using namespace std;

Dataset::Dataset(string filename) {
    readCSV(filename);
    preprocess();
}

Dataset::~Dataset() {
}

void Dataset::readCSV(string filename) {
    ifstream file(filename);

    stringstream buffer;
    buffer << file.rdbuf();
    string point;
    while(getline(buffer, point, ',')) {
        data.push_back(stod(point));
    }

    cout << "CSV file: " << data.size() << " points loaded" << endl;
}

float Dataset::normalize(float n) {
    return (n + min) / span;
}

float Dataset::denormalize(float n) {
    return (n * span) - min;
}

void Dataset::preprocess() {
    //normalize points into a [0,1] range
    min = *min_element(data.begin(), data.end());
    max = *max_element(data.begin(), data.end());
    span = max - min;

    vector<float> normalized;
    for (float n : data) {
        normalized.push_back(normalize(n));
    }

    data = normalized;
}

pair<vector<float>, vector<float>> Dataset::get_batch_sliding_window(int window) {
    vector<float> x;
    vector<float> y;

    cout << "window size: " << window << " | data size: " << data.size() << endl;

    for (int i=0; i!=data.size() - window; i++) {
        for (int j=0; j!=window; j++) {
            x.push_back(data[i+j]);
        }

        y.push_back(data[i + window]);
    }

    cout << "x vector size: " << x.size() << endl;
    cout << "y vector size: " << y.size() << endl;

    return make_pair(x, y);
}