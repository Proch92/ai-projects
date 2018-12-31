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
    return (n - min) / span;
}

float Dataset::denormalize(float n) {
    return (n * span) + min;
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

vector<Dataset::Batch> Dataset::get_batches_sliding_window(int batch_size, int window_size) {
    vector<Dataset::Batch> batches;

    cout << "data size: " << data.size() << " | window_size: " << window_size << " | batch_size: " << batch_size << endl;
    int num_points = data.size() - window_size;
    cout << "num points: " << num_points << endl;
    int points_in_batch = 0;
    Dataset::Batch batch;
    for (int i=0; i!=num_points; i++) {
        for (int j=0; j != window_size; j++) {
            batch.x.push_back(data[i+j]);
        }

        batch.y.push_back(data[i + window_size]);

        points_in_batch++;
        if (points_in_batch == batch_size) {
            batches.push_back(batch);
            batch.x.clear();
            batch.y.clear();
            points_in_batch = 0;
        }
    }
    //check any left over
    if (points_in_batch != 0) {
        batches.push_back(batch);
    }

    cout << "number of batches: " << batches.size() << endl;
    for (auto batch : batches) {
        cout << "batch x size: " << batch.x.size() << endl;
        cout << "batch y size: " << batch.y.size() << endl;
        cout << "________________________________" << endl;
    }

    return batches;
}

vector<Dataset::Batch> Dataset::get_batches(int batch_size, int window_size) {
    vector<Dataset::Batch> batches;

    cout << "data size: " << data.size() << " | window_size: " << window_size << " | batch_size: " << batch_size << endl;
    int min_seq = (batch_size * window_size) + 1;
    cout << "minimum data points required in sequence ((ws*bs)+1): " << min_seq << endl;
    if (data.size() < min_seq) {
        cout << "data sequence has not enough points: data size: " << data.size() << " data required to form a batch: " << min_seq << endl;
        exit(2);
    }

    int num_batches = (data.size() - 1) / (window_size * batch_size);

    Dataset::Batch batch;
    for (int i=0; i!=num_batches; i++) {
        int base_index = i * window_size * batch_size;
        for (int j=0; j!=window_size*batch_size; j++) {
            batch.x.push_back(data[base_index + j]);
        }

        for (int j=1; j<=batch_size; j++) {
            batch.y.push_back(data[base_index + (window_size * j)]);
        }

        batches.push_back(batch);
    }

    cout << "number of batches: " << batches.size() << endl;
    for (auto batch : batches) {
        cout << "batch x size: " << batch.x.size() << endl;
        cout << "batch y size: " << batch.y.size() << endl;
        cout << "________________________________" << endl;
    }

    return batches;
}