#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"

#include "dataset.h"
#include "simplednn.h"
#include <utility>
#include <iostream>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

SimpleDNN::SimpleDNN() {
}

SimpleDNN::~SimpleDNN() {
}

void SimpleDNN::train(Dataset dataset, int epochs, int batch_size, int window_size) {
    cout << "training" << endl;

    auto scope = Scope::NewRootScope();

    //input layer
    auto x = Placeholder(scope, DT_FLOAT);
    auto y = Placeholder(scope, DT_FLOAT);

    auto w_hidden = Variable(scope, {window_size, 3}, DT_FLOAT);
    auto init_rand_w_hidden = Assign(scope, w_hidden, RandomNormal(scope, {window_size, 3}, DT_FLOAT));

    auto b_hidden = Variable(scope, {1, 3}, DT_FLOAT);
    auto init_rand_b_hidden = Assign(scope, b_hidden, RandomNormal(scope, {1, 3}, DT_FLOAT));
    
    auto w_out = Variable(scope, {3, 1}, DT_FLOAT);
    auto init_rand_w_out = Assign(scope, w_out, RandomNormal(scope, {3, 1}, DT_FLOAT));

    auto b_out = Variable(scope, {1, 1}, DT_FLOAT);
    auto init_rand_b_out = Assign(scope, b_out, RandomNormal(scope, {1, 1}, DT_FLOAT));

    //building the graph
    auto hidden = Tanh(scope, Add(scope, MatMul(scope, x, w_hidden), b_hidden));
    auto out = Tanh(scope, Add(scope, MatMul(scope, hidden, w_out), b_out));

    //regularization
    auto regularization = AddN(scope, initializer_list<Input> {L2Loss(scope, w_hidden), L2Loss(scope, w_out)});

    //loss function
    auto loss = Add(scope, ReduceMean(scope, Square(scope, Sub(scope, out, y)), {0, 1}), Mul(scope, Cast(scope, 0.01, DT_FLOAT), regularization));

    cout << "now starting backprop graph" << endl;

    //learning
    //calculate gradients
    vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w_hidden, w_out, b_hidden, b_out}, &grad_outputs));

    cout << "gradients made" << endl;

    //update weights
    auto apply_w_hidden = ApplyGradientDescent(scope, w_hidden, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[0]});
    auto apply_w_out = ApplyGradientDescent(scope, w_out, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[1]});
    auto apply_b_hidden = ApplyGradientDescent(scope, b_hidden, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[2]});
    auto apply_b_out = ApplyGradientDescent(scope, b_out, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[3]});
    
    cout << "graph created" << endl;

    ClientSession session(scope);

    vector<Dataset::Batch> batches = dataset.get_batches_sliding_window(batch_size, window_size);

    //start training cycle
    cout << "randomly initializing weigths and bias..." << endl;
    TF_CHECK_OK(session.Run({init_rand_w_hidden, init_rand_w_out, init_rand_b_hidden, init_rand_b_out}, nullptr));
    cout << "...done" << endl;

    vector<Tensor> output;
    for (int i=0; i!=epochs; i++) {
        for (auto batch : batches) {
            Tensor tensor_x(DT_FLOAT, TensorShape{batch.y.size(), window_size});
            Tensor tensor_y(DT_FLOAT, TensorShape{batch.y.size(), 1});
            copy_n(batch.x.begin(), batch.x.size(), tensor_x.flat<float>().data());
            copy_n(batch.y.begin(), batch.y.size(), tensor_y.flat<float>().data());

            if (i%100 == 0) {
                TF_CHECK_OK(session.Run({{x, tensor_x}, {y, tensor_y}}, {loss}, &output));
                cout << "loss after " << i << " steps: " << output[0].scalar<float>() << endl;
            }

            TF_CHECK_OK(session.Run({{x, tensor_x}, {y, tensor_y}}, {apply_w_hidden, apply_w_out, apply_b_hidden, apply_b_out}, nullptr));
        }
    }
}