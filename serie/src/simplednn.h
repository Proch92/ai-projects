#include "tensorflow/cc/ops/standard_ops.h"

class SimpleDNN
{
public:
    SimpleDNN();
    ~SimpleDNN();
    void train(Dataset, int epochs, int batch_size, int window_size);
private:
};