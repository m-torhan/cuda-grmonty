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
    name = "harm_data_test",
    size = "small",
    srcs = ["harm_data_test.cpp"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "//cuda_grmonty:harm_data",
        "//cuda_grmonty:ndarray",
    ],
)
