# Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
#
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:defs.bzl", "cc_binary")

package_group(
    name = "cuda_grmonty_internal",
    packages = ["//..."],
)

package(
    default_visibility = ["//cuda_grmonty:cuda_grmonty_internal"],
)

cc_library(
    name = "consts",
    hdrs = ["consts.hpp"],
)

cc_library(
    name = "monty_rand",
    srcs = ["monty_rand.cpp"],
    hdrs = ["monty_rand.hpp"],
)

cc_library(
    name = "proba",
    srcs = ["proba.cpp"],
    hdrs = ["proba.hpp"],
    deps = [
        ":consts",
        ":monty_rand",
    ],
)

cc_library(
    name = "linalg",
    hdrs = ["linalg.hpp"],
    deps = [
        ":consts",
    ],
)

cc_library(
    name = "ndarray",
    hdrs = ["ndarray.hpp"],
    deps = [
        ":consts",
        ":linalg",
    ],
)

cc_library(
    name = "tetrads",
    srcs = ["tetrads.cpp"],
    hdrs = ["tetrads.hpp"],
    deps = [
        ":consts",
        ":ndarray",
    ],
)

cc_library(
    name = "integration",
    srcs = ["integration.cpp"],
    hdrs = ["integration.hpp"],
)

cc_library(
    name = "mathfn",
    srcs = ["mathfn.cpp"],
    hdrs = ["mathfn.hpp"],
)

cc_library(
    name = "hotcross",
    srcs = ["hotcross.cpp"],
    hdrs = ["hotcross.hpp"],
    deps = [
        ":consts",
        ":mathfn",
        ":ndarray",
        "@spdlog//:spdlog",
    ],
)

cc_library(
    name = "jnu_mixed",
    srcs = ["jnu_mixed.cpp"],
    hdrs = ["jnu_mixed.hpp"],
    deps = [
        ":consts",
        ":integration",
        ":mathfn",
        ":ndarray",
        "@spdlog//:spdlog",
    ],
)

cc_library(
    name = "radiation",
    srcs = ["radiation.cpp"],
    hdrs = ["radiation.hpp"],
    deps = [
        ":consts",
        ":hotcross",
        ":jnu_mixed",
        ":ndarray",
        "@spdlog//:spdlog",
    ],
)

cc_library(
    name = "parse_verbosity",
    srcs = ["parse_verbosity.cpp"],
    hdrs = ["parse_verbosity.hpp"],
    deps = [
        "@spdlog//:spdlog",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_library(
    name = "harm_model",
    srcs = ["harm_model.cpp"],
    hdrs = ["harm_model.hpp"],
    deps = [
        ":consts",
        ":hotcross",
        ":jnu_mixed",
        ":monty_rand",
        ":ndarray",
        ":proba",
        ":radiation",
        ":tetrads",
        "@spdlog//:spdlog",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        ":harm_model",
        ":monty_rand",
        ":parse_verbosity",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@spdlog//:spdlog",
    ],
)
