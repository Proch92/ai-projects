#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <utility>
#include "dataset.h"

using namespace std;

Dataset::Dataset(string filename) {
    data = readCSV(filename);
}

Dataset::~Dataset() {

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

    cout << "CSV file: " << points.size() << " points loaded" << endl;

    return points;
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