cc_test(
    name = "ndarray_test",
    size = "small",
    srcs = ["ndarray_test.cpp"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "//cuda_grmonty:ndarray",
    ],
)

cc_test(
    name = "integration_test",
    size = "small",
    srcs = ["integration_test.cpp"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "//cuda_grmonty:integration",
    ],
)

cc_test(
    name = "harm_model_test",
    size = "small",
    srcs = ["harm_model_test.cpp"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "//cuda_grmonty:harm_model",
        "//cuda_grmonty:ndarray",
    ],
)
