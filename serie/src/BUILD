load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "main",
    srcs = ["main.cc",
            "dataset.cc",
            "dataset.h",
            "simplednn.cc",
            "simplednn.h",
            "rnn.cc",
            "rnn.h"],
    deps = [
        "@org_tensorflow//tensorflow/cc:gradients",
        "@org_tensorflow//tensorflow/cc:grad_ops",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow"
    ],
)
