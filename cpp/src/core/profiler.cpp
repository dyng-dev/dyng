// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file profiler.cpp
 * @brief The instance-based profiler.
 */
#include "core/cuda_runtime.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>

#include <cstdio>
#include <exception>
#include <ostream>

namespace dyng {

namespace {

bool is_name_start(char c) noexcept {
  return c >= 'a' && c <= 'z';
}

bool is_name_char(char c) noexcept {
  return is_name_start(c) || (c >= '0' && c <= '9') || c == '_';
}

std::string format_ms(double ms) {
  char text[64];
  std::snprintf(text, sizeof(text), "%.6f", ms);
  return text;
}

}  // namespace

profiler::profiler(profiler_options options) : options_(options) {}

bool profiler::is_valid_name(std::string_view name) noexcept {
  int parts = 0;
  bool at_part_start = true;
  for (char c : name) {
    if (at_part_start) {
      if (!is_name_start(c)) {
        return false;
      }
      at_part_start = false;
      ++parts;
    } else if (c == '.') {
      at_part_start = true;
    } else if (!is_name_char(c)) {
      return false;
    }
  }
  return !at_part_start && parts >= 2;
}

void profiler::begin_stage(std::string_view name) {
  DYNG_EXPECTS(is_valid_name(name), "profiler stage name '", name,
               "' does not follow <algo>.<hook>[.<sub>] in dotted lower case");
  std::size_t index = stages_.size();
  for (std::size_t i = 0; i < stages_.size(); ++i) {
    if (stages_[i].name == name) {
      index = i;
      break;
    }
  }
  if (index == stages_.size()) {
    stage_record record;
    record.name = std::string(name);
    record.depth = static_cast<int>(open_.size());
    stages_.push_back(std::move(record));
  }
  open_.push_back(open_stage{index, std::chrono::steady_clock::now()});
}

void profiler::end_stage(double device_ms) {
  const auto now = std::chrono::steady_clock::now();
  DYNG_EXPECTS(!open_.empty(), "profiler::end_stage() without an open stage");
  const open_stage top = open_.back();
  open_.pop_back();
  const double host_ms = std::chrono::duration<double, std::milli>(now - top.start).count();
  stage_record& record = stages_[top.record];
  record.calls += 1;
  record.host_ms += host_ms;
  record.device_ms += device_ms;
  samples_.push_back(stage_sample{record.name, static_cast<int>(open_.size()), host_ms, device_ms});
}

void profiler::add_counter(std::string_view name, std::int64_t value) {
  DYNG_EXPECTS(is_valid_name(name), "profiler counter name '", name,
               "' does not follow <algo>.<name>[.<sub>] in dotted lower case");
  for (auto& c : counters_) {
    if (c.name == name) {
      c.value += value;
      return;
    }
  }
  counters_.push_back(counter_record{std::string(name), value});
}

double profiler::total_host_ms(std::string_view name) const noexcept {
  for (const auto& s : stages_) {
    if (s.name == name) {
      return s.host_ms;
    }
  }
  return 0.0;
}

std::int64_t profiler::counter(std::string_view name) const noexcept {
  for (const auto& c : counters_) {
    if (c.name == name) {
      return c.value;
    }
  }
  return 0;
}

void profiler::reset() {
  DYNG_EXPECTS(open_.empty(), "profiler::reset() while ", open_.size(), " stage(s) are open");
  stages_.clear();
  samples_.clear();
  counters_.clear();
}

void profiler::write_csv(std::ostream& out) const {
  out << "kind,name,value\n";
  for (const auto& s : samples_) {
    out << "stage," << s.name << ',' << format_ms(s.host_ms) << '\n';
  }
  for (const auto& c : counters_) {
    out << "counter," << c.name << ',' << c.value << '\n';
  }
}

void profiler::write_json(std::ostream& out) const {
  // Names are validated ([a-z0-9_.]), so they need no escaping.
  out << "{\"stages\": [";
  for (std::size_t i = 0; i < stages_.size(); ++i) {
    const auto& s = stages_[i];
    out << (i == 0 ? "" : ", ") << "{\"name\": \"" << s.name << "\", \"depth\": " << s.depth
        << ", \"calls\": " << s.calls << ", \"host_ms\": " << format_ms(s.host_ms)
        << ", \"device_ms\": " << format_ms(s.device_ms) << '}';
  }
  out << "], \"counters\": [";
  for (std::size_t i = 0; i < counters_.size(); ++i) {
    out << (i == 0 ? "" : ", ") << "{\"name\": \"" << counters_[i].name
        << "\", \"value\": " << counters_[i].value << '}';
  }
  out << "]}\n";
}

// ----------------------------------------------------------------------------------------------
// scoped_stage
// ----------------------------------------------------------------------------------------------

scoped_stage::scoped_stage(const resources& res, std::string_view name)
    : res_(&res), profiler_(res.get_profiler()) {
  if (profiler_ != nullptr) {
    if (profiler_->options().sync_stages) {
      res.synchronize();
    }
    profiler_->begin_stage(name);
    if (profiler_->options().cuda_events && res.get_backend() == backend::cuda) {
      try {
        events_ = std::make_unique<detail::cuda_event_timer>(res.device(), res.stream());
      } catch (...) {
        profiler_->end_stage();  // keep the profiler's stage stack balanced
        profiler_ = nullptr;
        throw;
      }
    }
  }
}

scoped_stage::scoped_stage(profiler* p, std::string_view name) : profiler_(p) {
  if (profiler_ != nullptr) {
    profiler_->begin_stage(name);
  }
}

scoped_stage::~scoped_stage() {
  stop();
}

void scoped_stage::stop() noexcept {
  if (profiler_ == nullptr) {
    return;
  }
  profiler* p = profiler_;
  profiler_ = nullptr;
  double device_ms = 0.0;
  bool balanced = false;
  try {
    if (events_ != nullptr) {
      device_ms = events_->stop();
    }
    if (res_ != nullptr && p->options().sync_stages) {
      res_->synchronize();
    }
    balanced = true;
    p->end_stage(device_ms);
  } catch (const std::exception& e) {
    // Never throw from a destructor path; a failed synchronization resurfaces on the next call.
    log_message(log_level::error, e.what());
  } catch (...) {
    log_message(log_level::error, "dyng: unknown exception while ending a profiler stage");
  }
  events_.reset();
  if (!balanced) {
    try {
      p->end_stage(device_ms);  // keep the stage stack balanced after a failed wait
    } catch (...) {
      log_message(log_level::error, "dyng: could not end a profiler stage");
    }
  }
}

}  // namespace dyng
