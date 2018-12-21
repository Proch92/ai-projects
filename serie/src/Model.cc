#include "dataset.h"
#include "model.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

SimpleDNN::SimpleDNN() {
    Scope scope = Scope::NewRootScope();

    //input layer
    auto x = Placeholder(scope, DT_FLOAT);
    auto y = Placeholder(scope, DT_FLOAT);
    //hidden layer
    auto w_hidden = Variable(scope, {3, 3}, DT_FLOAT);
    auto b_hidden = Variable(scope, {1, 3}, DT_FLOAT);
    //out_layer
    auto w_out = Variable(scope, {3, 1}, DT_FLOAT);
    auto b_out = Variable(scope, {1, 1}, DT_FLOAT);

    //building the graph
    auto hidden = Relu(scope, BiasAdd(scope, MatMul(scope, w_hidden, x), b_hidden));
    auto out = Relu(scope, BiasAdd(scope, MatMul(scope, w_out, hidden), b_out));

    //regularization
    auto regularization = AddN(scope, initializer_list<Input> {L2Loss(scope, w_hidden), L2Loss(scope, w_out)});

    //loss function
    auto loss = Sub(scope, out, y);
}

void SimpleDNN::train(Dataset dataset) {

}