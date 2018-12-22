#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "dataset.h"
#include "model.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

SimpleDNN::SimpleDNN() {
    scope = Scope::NewRootScope();

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
    auto loss = Add(scope, Sub(scope, out, y), Mul(scope, Cast(scope, 0.01, DT_FLOAT), regularization));

    //learning
    //calculate gradients
    vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w_hidden, w_out, b_hidden, b_out}, &grad_outputs));

    //update weights
    auto apply_w_hidden = ApplyGradientDescent(scope, w_hidden, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[0]});
    auto apply_b_hidden = ApplyGradientDescent(scope, b_hidden, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[1]});
    auto apply_w_out = ApplyGradientDescent(scope, w_out, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[2]});
    auto apply_b_out = ApplyGradientDescent(scope, b_out, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[3]});

    //load module or init the weights
    auto init_rand_w_hidden = Assign(scope, w_hidden, RandomNormal(scope, {3,3}, DT_FLOAT));
    auto init_rand_w_out = Assign(scope, w_out, RandomNormal(scope, {3,1}, DT_FLOAT));
    auto init_rand_b_hidden = Assign(scope, b_hidden, RandomNormal(scope, {1,3}, DT_FLOAT));
    auto init_rand_b_out = Assign(scope, b_out, RandomNormal(scope, {1,1}, DT_FLOAT));
}

SimpleDNN::~SimpleDNN() {
}

void SimpleDNN::train(Dataset dataset, int epochs) {
    ClientSession session(scope);
    TF_CHECK_OK(session.Run({init_rand_w_hidden, init_rand_w_out, init_rand_b_hidden, init_rand_b_out}, nullptr));

    vector<Tensor> output;
    for (int i=0; i!=epochs; i++) {
        if (i%100 == 0) {
            TF_CHECK_OK(session.Run({{x, data_x}, {y, data_y}}, {loss}, &output));
            cout << "loss after " << i << "steps: " << output[0],scalar<float>() << endl;
        }

        TF_CHECK_OK(session.Run({{x, data_x}, {y, data_y}}, {apply_w_hidden, apply_w_out, apply_b_hidden, apply_b_out}, nullptr));
    }
}