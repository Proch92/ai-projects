#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"

#include "dataset.h"
#include "rnn.h"
#include <utility>
#include <iostream>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

Rnn::Rnn() {
}

Rnn::~Rnn() {
}

void Rnn::train(Dataset dataset, int epochs) {
    cout << "building graph" << endl;
    auto scope = Scope::NewRootScope();

    //input layer
    auto x = Placeholder(scope, DT_FLOAT);
    auto y = Placeholder(scope, DT_FLOAT);

    // RNN variables and inits
    auto w_rnn = Variable(scope, {state_size+1, state_size}, DT_FLOAT);
    auto init_rand_w_rnn = Assign(scope, w_rnn, RandomNormal(scope, {state_size+1, state_size}, DT_FLOAT));
    auto b_rnn = Variable(scope, {1, state_size}, DT_FLOAT);
    auto init_rand_b_rnn = Assign(scope, b_rnn, RandomNormal(scope, {1, state_size}, DT_FLOAT));

    // Dense out
    auto w_dense = Variable(scope, {state_size, 1});
    auto init_rand_w_dense = Assign(scope, w_dense, RandomNormal(scope, {state_size, 1}, DT_FLOAT));
    auto b_dense = Variable(scope, {1, 1});
    auto init_rand_b_dense = Assign(scope, b_dense, RandomNormal(scope, {1, 1}, DT_FLOAT));

    // slice input into window_size slices of shape {batch_size, 1}
    OutputList input_slices = Unpack(scope, x, 1);

    auto state = ZerosLike(scope, {batch_size, state_size});

    for (int i=0; i!=window_size; i++) {
        auto concat_state_input = Concat(scope, {input_slices[i], state}, 1);
        state = Tanh(scope, Add(scope, MatMul(scope, concat_state_input, w_rnn), b_rnn));
    }

    // dense output
    auto out = Tanh(scope, Add(scope, MatMul(scope, state, w_dense), b_dense));

    // loss function
    auto loss = ReduceMean(scope, Square(scope, Sub(scope, out, y)), {0, 1});

    cout << "now starting backprop graph" << endl;

    //learning
    //calculate gradients
    vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w_rnn, w_dense, b_rnn, b_dense}, &grad_outputs));

    cout << "gradients made" << endl;

    //update weights
    auto apply_w_rnn = ApplyGradientDescent(scope, w_hidden, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[0]});
    auto apply_w_dense = ApplyGradientDescent(scope, w_out, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[1]});
    auto apply_b_rnn = ApplyGradientDescent(scope, b_hidden, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[2]});
    auto apply_b_dense = ApplyGradientDescent(scope, b_out, Cast(scope, 0.01, DT_FLOAT), {grad_outputs[3]});
    
    cout << "graph created" << endl;

    ClientSession session(scope);

    //load data into input tensors
    cout << "loading data into tensors..." << endl;
    pair<vector<float>, vector<float>> batch = dataset.get_batch_sliding_window(window_size);
    int batch_size = batch.first.size() / window_size;
    cout << "...done. batch size = " << batch_size << endl;

    Tensor data_x(DT_FLOAT, TensorShape{batch_size, window_size});
    Tensor data_y(DT_FLOAT, TensorShape{batch_size, 1});
    copy_n(batch.first.begin(), batch.first.size(), data_x.flat<float>().data());
    copy_n(batch.second.begin(), batch.second.size(), data_y.flat<float>().data());

    //start training cycle
    cout << "randomly initializing weigths and bias..." << endl;
    TF_CHECK_OK(session.Run({init_rand_w_rnn, init_rand_w_dense, init_rand_b_rnn, init_rand_b_dense}, nullptr));
    cout << "...done" << endl;

    vector<Tensor> output;
    for (int i=0; i!=epochs; i++) {
        if (i%100 == 0) {
            TF_CHECK_OK(session.Run({{x, data_x}, {y, data_y}}, {loss}, &output));
            cout << "loss after " << i << " steps: " << output[0].scalar<float>() << endl;
        }

        TF_CHECK_OK(session.Run({{x, data_x}, {y, data_y}}, {apply_w_rnn, apply_w_dense, apply_b_rnn, apply_b_dense}, nullptr));
    }
}