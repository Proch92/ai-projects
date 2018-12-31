#include <vector>
#include <string>
#include <utility>

using namespace std;

class Dataset
{
public:
    struct Batch {
        vector<float> x;
        vector<float> y;
    };

    Dataset(string);
    ~Dataset();

    vector<float> get_plain_data() {return data;}
    vector<Batch> get_batches_sliding_window(int batch_size, int window_size);
    vector<Batch> get_batches(int batch_size, int window_size);

private:
    void readCSV(string);
    void preprocess();
    float normalize(float);
    float denormalize(float);

    vector<float> data;
    float min, max, span;
};