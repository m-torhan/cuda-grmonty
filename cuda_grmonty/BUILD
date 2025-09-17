# Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
#
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_cuda//cuda:defs.bzl", "cuda_library", "requires_cuda")

package_group(
    name = "cuda_grmonty_internal",
    packages = ["//..."],
)

package(
    default_visibility = ["//cuda_grmonty:cuda_grmonty_internal"],
)

config_setting(
    name = "cuda_enabled",
    values = {"define": "CUDA=true"},
)

cuda_library(
    name = "cuda_utils",
    hdrs = ["utils.cuh"],
    target_compatible_with = requires_cuda(),
)

cc_library(
    name = "consts",
    hdrs = ["consts.hpp"],
)

cc_library(
    name = "photon",
    hdrs = ["photon.hpp"],
    deps = [
        ":consts",
    ],
)

cc_library(
    name = "photon_queue",
    srcs = ["photon_queue.cpp"],
    hdrs = ["photon_queue.hpp"],
    deps = [
        ":photon",
    ],
)

cc_library(
    name = "harm_data",
    hdrs = ["harm_data.hpp"],
    deps = [
        ":ndarray",
    ],
)

cc_library(
    name = "cuda_harm_data",
    hdrs = ["harm_data.cuh"],
    target_compatible_with = requires_cuda(),
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

cuda_library(
    name = "cuda_proba",
    hdrs = ["proba.cuh"],
    deps = [
        ":consts",
    ],
    target_compatible_with = requires_cuda(),
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

cuda_library(
    name = "cuda_tetrads",
    hdrs = ["tetrads.cuh"],
    deps = [
        ":consts",
    ],
    target_compatible_with = requires_cuda(),
)

cc_library(
    name = "integration",
    srcs = ["integration.cpp"],
    hdrs = ["integration.hpp"],
)

cuda_library(
    name = "cuda_hotcross",
    hdrs = ["hotcross.cuh"],
    deps = [
        ":consts",
    ],
    target_compatible_with = requires_cuda(),
)

cuda_library(
    name = "cuda_hotcross_table",
    srcs = ["hotcross_table.cu"],
    hdrs = ["hotcross_table.cuh"],
    deps = [
        ":consts",
        ":cuda_hotcross",
        ":cuda_utils",
        ":ndarray",
    ],
    target_compatible_with = requires_cuda(),
)

cc_library(
    name = "hotcross",
    srcs = ["hotcross.cpp"],
    hdrs = ["hotcross.hpp"],
    deps = [
        ":consts",
        ":ndarray",
        "@spdlog//:spdlog",
    ] + select({
        ":cuda_enabled": [":cuda_hotcross_table"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "jnu_mixed",
    srcs = ["jnu_mixed.cpp"],
    hdrs = ["jnu_mixed.hpp"],
    deps = [
        ":consts",
        ":integration",
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

cuda_library(
    name = "cuda_jnu_mixed",
    hdrs = ["jnu_mixed.cuh"],
    deps = [
        ":consts",
    ],
    target_compatible_with = requires_cuda(),
)

cuda_library(
    name = "cuda_radiation",
    hdrs = ["radiation.cuh"],
    deps = [
        ":consts",
        ":cuda_hotcross",
        ":cuda_jnu_mixed",
    ],
    target_compatible_with = requires_cuda(),
)

cuda_library(
    name = "cuda_harm_model",
    hdrs = ["harm_model.cuh"],
    deps = [
        ":consts",
        ":cuda_harm_data",
        ":cuda_tetrads",
        ":cuda_utils",
        ":harm_data",
    ],
    target_compatible_with = requires_cuda(),
)

cuda_library(
    name = "cuda_super_photon",
    srcs = ["super_photon.cu"],
    hdrs = ["super_photon.cuh"],
    deps = [
        ":cuda_harm_data",
        ":cuda_harm_model",
        ":cuda_proba",
        ":cuda_radiation",
        ":cuda_utils",
        ":harm_data",
        ":photon",
        ":photon_queue",
    ],
    target_compatible_with = requires_cuda(),
)

cc_library(
    name = "harm_model",
    srcs = ["harm_model.cpp"],
    hdrs = ["harm_model.hpp"],
    deps = [
        ":consts",
        ":harm_data",
        ":hotcross",
        ":jnu_mixed",
        ":monty_rand",
        ":ndarray",
        ":photon",
        ":photon_queue",
        ":proba",
        ":radiation",
        ":tetrads",
        "@spdlog//:spdlog",
    ] + select({
        ":cuda_enabled": [":cuda_super_photon"],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        ":consts",
        ":harm_model",
        ":monty_rand",
        ":photon",
        ":parse_verbosity",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@spdlog//:spdlog",
    ],
)
