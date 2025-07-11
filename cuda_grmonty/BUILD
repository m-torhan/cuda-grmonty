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
    hdrs = ["consts.hpp"]
)

cc_library(
    name = "ndarray",
    hdrs = ["ndarray.hpp"],
    deps = [
        ":consts",
        ":linalg",
    ]
)

cc_library(
    name = "parse_verbosity",
    srcs = ["parse_verbosity.cpp"],
    hdrs = ["parse_verbosity.hpp"],
    deps = [
        "@spdlog//:spdlog",
        "@abseil-cpp//absl/strings:string_view",
    ]
)

cc_library(
    name = "harm_data",
    srcs = ["harm_data.cpp"],
    hdrs = ["harm_data.hpp"],
    deps = [
        ":consts",
        ":ndarray",
        "@spdlog//:spdlog",
    ]
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        ":parse_verbosity",
        ":harm_data",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@spdlog//:spdlog",
    ]
)
