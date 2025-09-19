/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "absl/strings/string_view.h"
#include "spdlog/common.h"

namespace spdlog::level {

/**
 * @brief Parse a logging verbosity level from text.
 *
 * @param[in] text   Logging verbosity in text format (e.g., "info", "debug").
 * @param[out] level Output parameter storing the parsed verbosity level.
 * @param[out] error Output parameter storing an error message if parsing fails.
 *
 * @returns True if parsing verbosity was succesfull, false otherwise.
 */
bool AbslParseFlag(absl::string_view text, level_enum *level, std::string *error);

/**
 * Converts spdlog level enum to string form.
 *
 * @param level Logging verbosity level num.
 *
 * @returns Logging verbosity as string (e.g., "info", "debug").
 */
std::string AbslUnparseFlag(level_enum level);

}; /* namespace spdlog::level */
