using namespace std;

public class Dataset
{
public:
    Dataset(string filename);
    ~Dataset();

    vector<float> get_data() {return x;}

private:
    vector<float> readCSV(string filename);
    vector<float> x;
};