#include <vector>
#include <string>
#include <utility>

using namespace std;

class Dataset
{
public:
    Dataset(string);
    ~Dataset();

    vector<float> get_plain_data() {return data;}
    pair<vector<float>, vector<float>> get_batch_sliding_window(int);

private:
    vector<float> readCSV(string);
    vector<float> data;
};