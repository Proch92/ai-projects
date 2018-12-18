#include "Dataset.h"
#include "Model.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

void Model::Model() {
    Scope scope = Scope::NewRootScope();

    //input layer
    auto x = Placeholder(scope, DT_FLOAT);
    //hidden layer
    auto w_hidden = Variable(scope, {3, 3}, DT_FLOAT);
    auto b_hidden = Variable(scope, {1, 3}, DT_FLOAT);
    //out_layer
    auto w_out = Variable(scope, {3, 1}, DT_FLOAT);
    auto b_out = Variable(scope, {1, 1}, DT_FLOAT);

    //building the graph
    auto hidden = Relu(scope, BiasAdd(scope, MatMul(w_hidden, x), b_hidden));
    auto out = Relu(scope, BiasAdd(scope, MatMul(w_out, hidden), b_out));

    //regularization
    auto regularization = AddN(scope, {L2Loss(scope, w_hidden), L2Loss(scope, w_out)});
}

void Model::train(Dataset dataset) {

}