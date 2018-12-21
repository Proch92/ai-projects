#include <vector>
#include <string>

using namespace std;

class Dataset
{
public:
    Dataset(string);
    ~Dataset();

    vector<float> get_data() {return x;}

private:
    vector<float> readCSV(string);
    vector<float> x;
};