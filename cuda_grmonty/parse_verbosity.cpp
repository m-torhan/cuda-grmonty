/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include "cuda_grmonty/parse_verbosity.hpp"

namespace spdlog::level {

bool AbslParseFlag(absl::string_view text, level_enum *level, std::string *error) {
    if (text == "trace") {
        *level = trace;
        return true;
    }
    if (text == "debug") {
        *level = debug;
        return true;
    }
    if (text == "info") {
        *level = info;
        return true;
    }
    if (text == "warn") {
        *level = warn;
        return true;
    }
    if (text == "err") {
        *level = err;
        return true;
    }
    if (text == "critical") {
        *level = critical;
        return true;
    }
    if (text == "off") {
        *level = off;
        return true;
    }
    *error = fmt::format("Invalid verbosity {}", text);
    return false;
}

std::string AbslUnparseFlag(level_enum level) {
    switch (level) {
    case trace:
        return "trace";
    case debug:
        return "debug";
    case info:
        return "info";
    case warn:
        return "warn";
    case err:
        return "err";
    case critical:
        return "critical";
    case off:
        return "off";
    default:
        return "unknown";
    }
}

}; /* namespace spdlog::level */
