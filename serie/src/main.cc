#include <iostream>
#include <string>
#include "dataset.h"
#include "simplednn.h"
#include "rnn.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "usage: ./" << argv[0] << " csv_filename epochs" << endl;
        return 1;
    }

    printf("loading dataset\n");
    string filename(argv[1]);
    Dataset dataset(filename);

    SimpleDNN dnn;
    dnn.train(dataset, atoi(argv[2]), 5, 3);

    Rnn rnn;
    rnn.train(dataset, atoi(argv[2]), 3, 3);

    return 0;
}
