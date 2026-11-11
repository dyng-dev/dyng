// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/checker.cu (DynLPChecker::harmonicSolution, checkLabels)
/**
 * @file harmonic_solution.cpp
 * @brief The harmonic oracle (see harmonic_solution.hpp): LabelPropagation-CUDA's PCG with
 *        reductions in a fixed order. Standard library only.
 */
#include <dyng/testing/harmonic_solution.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace dyng::testing {

namespace {

/// Vertices per reduction chunk: partial sums per chunk, then the chunks in order.
constexpr std::size_t reduce_chunk = 4096;
/// Below this many vertices per thread the work runs on the calling thread.
constexpr std::size_t min_per_thread = 50000;

/// Runs fn(first_chunk, last_chunk) over [0, num_chunks) on up to `threads` threads.
template <typename fn_t>
void for_chunks(std::size_t num_chunks, std::size_t n, int threads, const fn_t& fn) {
  const std::size_t want = std::min<std::size_t>(static_cast<std::size_t>(std::max(threads, 1)),
                                                 std::max<std::size_t>(n / min_per_thread, 1));
  const std::size_t t = std::min(want, std::max<std::size_t>(num_chunks, 1));
  if (t <= 1) {
    fn(std::size_t{0}, num_chunks);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(t - 1);
  const std::size_t per = (num_chunks + t - 1) / t;
  for (std::size_t k = 1; k < t; ++k) {
    const std::size_t first = std::min(num_chunks, k * per);
    const std::size_t last = std::min(num_chunks, first + per);
    pool.emplace_back([&fn, first, last] { fn(first, last); });
  }
  fn(std::size_t{0}, std::min(num_chunks, per));
  for (std::thread& th : pool) {
    th.join();
  }
}

/// Elementwise work fn(i) for i in [0, n).
template <typename fn_t>
void for_each_vertex(std::size_t n, int threads, const fn_t& fn) {
  const std::size_t chunks = (n + reduce_chunk - 1) / reduce_chunk;
  for_chunks(chunks, n, threads, [&](std::size_t c0, std::size_t c1) {
    const std::size_t end = std::min(n, c1 * reduce_chunk);
    for (std::size_t i = c0 * reduce_chunk; i < end; ++i) {
      fn(i);
    }
  });
}

/// sum_i a[i] * b[i] in a fixed order: per chunk in index order, then the chunks in order.
double fixed_dot(const std::vector<double>& a, const std::vector<double>& b, int threads,
                 std::vector<double>& partial) {
  const std::size_t n = a.size();
  const std::size_t chunks = (n + reduce_chunk - 1) / reduce_chunk;
  partial.assign(chunks, 0.0);
  for_chunks(chunks, n, threads, [&](std::size_t c0, std::size_t c1) {
    for (std::size_t c = c0; c < c1; ++c) {
      double s = 0.0;
      const std::size_t end = std::min(n, (c + 1) * reduce_chunk);
      for (std::size_t i = c * reduce_chunk; i < end; ++i) {
        s += a[i] * b[i];
      }
      partial[c] = s;
    }
  });
  double total = 0.0;
  for (const double s : partial) {
    total += s;
  }
  return total;
}

void check_sizes(std::size_t n, const std::vector<std::uint8_t>& present,
                 const std::vector<std::int8_t>& seed_labels, const char* what) {
  if (present.size() != n || seed_labels.size() != n) {
    throw std::invalid_argument(std::string(what) + ": present and seed_labels need one entry " +
                                "per vertex (" + std::to_string(n) + ")");
  }
}

}  // namespace

template <typename edge_t>
harmonic_result harmonic_solution(const std::vector<edge_t>& row_ptr,
                                  const std::vector<std::int32_t>& col_ind,
                                  const std::vector<float>& weights,
                                  const std::vector<std::uint8_t>& present,
                                  const std::vector<std::int8_t>& seed_labels,
                                  const harmonic_options& opt) {
  if (row_ptr.empty() || col_ind.size() != weights.size() ||
      static_cast<std::size_t>(row_ptr.back()) > col_ind.size()) {
    throw std::invalid_argument("harmonic_solution: inconsistent CSR arrays");
  }
  const std::size_t n = row_ptr.size() - 1;
  check_sizes(n, present, seed_labels, "harmonic_solution");
  if (!(opt.relative_tolerance >= 0.0) || opt.max_iterations <= 0 || opt.num_threads < 1) {
    throw std::invalid_argument("harmonic_solution: invalid options");
  }
  for (std::size_t v = 0; v < n; ++v) {
    if (seed_labels[v] < -1 || seed_labels[v] > 1) {
      throw std::invalid_argument("harmonic_solution: seed label of vertex " + std::to_string(v) +
                                  " is not -1, 0 or 1");
    }
    if (row_ptr[v] > row_ptr[v + 1]) {
      throw std::invalid_argument("harmonic_solution: decreasing row offsets");
    }
  }
  for (const std::int32_t c : col_ind) {
    if (c >= 0 && static_cast<std::size_t>(c) >= n) {
      throw std::invalid_argument("harmonic_solution: column " + std::to_string(c) +
                                  " out of range");
    }
  }
  const auto is_seed = [&](std::size_t v) { return present[v] != 0 && seed_labels[v] >= 0; };
  const auto is_unknown = [&](std::size_t v) { return present[v] != 0 && seed_labels[v] < 0; };
  const auto row = [&](std::size_t u) {
    return std::pair<std::size_t, std::size_t>(static_cast<std::size_t>(row_ptr[u]),
                                               static_cast<std::size_t>(row_ptr[u + 1]));
  };

  harmonic_result out;
  std::vector<std::uint8_t>& determined = out.determined;
  determined.assign(n, 0);

  // Components with a seed: multi-source BFS from the present seeds over unknown vertices.
  std::vector<std::size_t> queue;
  for (std::size_t v = 0; v < n; ++v) {
    if (is_seed(v)) {
      queue.push_back(v);
      ++out.num_seeds;
    } else if (is_unknown(v)) {
      ++out.num_unlabelled;
    }
  }
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const auto [b, e] = row(queue[head]);
    for (std::size_t k = b; k < e; ++k) {
      const std::int32_t c = col_ind[k];
      if (c < 0) {
        continue;
      }
      const auto v = static_cast<std::size_t>(c);
      if (is_unknown(v) && determined[v] == 0) {
        determined[v] = 1;
        queue.push_back(v);
      }
    }
  }
  for (std::size_t v = 0; v < n; ++v) {
    out.num_determined += determined[v];
  }
  out.num_excluded = out.num_unlabelled - out.num_determined;

  // The system on the unknowns: (D - W_UU) x = W_UL y_L.
  const int threads = opt.num_threads;
  std::vector<double> diag(n, 0.0);
  std::vector<double> rhs(n, 0.0);
  for_each_vertex(n, threads, [&](std::size_t u) {
    if (determined[u] == 0) {
      return;
    }
    double d = 0.0;
    double r = 0.0;
    const auto [b, e] = row(u);
    for (std::size_t k = b; k < e; ++k) {
      const std::int32_t c = col_ind[k];
      if (c < 0 || present[static_cast<std::size_t>(c)] == 0) {
        continue;
      }
      d += weights[k];
      if (seed_labels[static_cast<std::size_t>(c)] >= 0) {
        r += weights[k] * static_cast<float>(seed_labels[static_cast<std::size_t>(c)]);
      }
    }
    diag[u] = d;
    rhs[u] = r;
  });
  const auto matvec = [&](const std::vector<double>& in, std::vector<double>& res) {
    for_each_vertex(n, threads, [&](std::size_t u) {
      if (determined[u] == 0) {
        res[u] = 0.0;
        return;
      }
      double s = diag[u] * in[u];
      const auto [b, e] = row(u);
      for (std::size_t k = b; k < e; ++k) {
        const std::int32_t c = col_ind[k];
        if (c >= 0 && determined[static_cast<std::size_t>(c)] != 0) {
          s -= weights[k] * in[static_cast<std::size_t>(c)];
        }
      }
      res[u] = s;
    });
  };

  // Jacobi-preconditioned conjugate gradients from x0 = 0.
  std::vector<double>& x = out.solution;
  x.assign(n, 0.0);
  std::vector<double> r = rhs;
  std::vector<double> z(n, 0.0);
  std::vector<double> ap(n, 0.0);
  std::vector<double> partial;
  for (std::size_t u = 0; u < n; ++u) {
    if (determined[u] != 0) {
      z[u] = r[u] / diag[u];
    }
  }
  std::vector<double> p = z;
  double rz = fixed_dot(r, z, threads, partial);
  const double norm_b = std::sqrt(fixed_dot(rhs, rhs, threads, partial));
  const double stop = opt.relative_tolerance * norm_b;
  std::int64_t it = 0;
  for (; it < opt.max_iterations && std::sqrt(fixed_dot(r, r, threads, partial)) > stop; ++it) {
    matvec(p, ap);
    const double alpha = rz / fixed_dot(p, ap, threads, partial);
    for_each_vertex(n, threads, [&](std::size_t u) {
      x[u] += alpha * p[u];
      r[u] -= alpha * ap[u];
      z[u] = determined[u] != 0 ? r[u] / diag[u] : 0.0;
    });
    const double rz_new = fixed_dot(r, z, threads, partial);
    const double beta = rz_new / rz;
    rz = rz_new;
    for_each_vertex(n, threads, [&](std::size_t u) { p[u] = z[u] + beta * p[u]; });
  }
  out.iterations = it;
  out.relative_residual =
      norm_b > 0.0 ? std::sqrt(fixed_dot(r, r, threads, partial)) / norm_b : 0.0;
  out.converged = out.relative_residual <= opt.relative_tolerance;
  for (std::size_t v = 0; v < n; ++v) {
    if (is_seed(v)) {
      x[v] = seed_labels[v];
    } else if (determined[v] == 0) {
      x[v] = 0.5;
    }
  }
  return out;
}

label_error label_error_against(const std::vector<float>& labels, const harmonic_result& f,
                                const std::vector<std::uint8_t>& present,
                                const std::vector<std::int8_t>& seed_labels) {
  const std::size_t n = labels.size();
  check_sizes(n, present, seed_labels, "label_error_against");
  if (f.solution.size() != n || f.determined.size() != n) {
    throw std::invalid_argument("label_error_against: the solution is of another graph");
  }
  label_error out;
  double sum = 0.0;
  for (std::size_t v = 0; v < n; ++v) {
    if (present[v] == 0 || seed_labels[v] >= 0) {
      continue;
    }
    if (f.determined[v] == 0) {
      ++out.excluded;
      continue;
    }
    const double err = std::fabs(labels[v] - f.solution[v]);
    sum += err;
    out.max = std::fmax(out.max, err);
    if ((labels[v] >= 0.5F) != (f.solution[v] >= 0.5)) {
      ++out.flips;
    }
    ++out.compared;
  }
  out.mean = out.compared > 0 ? sum / static_cast<double>(out.compared) : 0.0;
  return out;
}

class_agreement class_agreement_with(const std::vector<float>& labels, const harmonic_result& f,
                                     const std::vector<std::uint8_t>& present,
                                     const std::vector<std::int8_t>& seed_labels, double margin) {
  const std::size_t n = labels.size();
  check_sizes(n, present, seed_labels, "class_agreement_with");
  if (f.solution.size() != n || f.determined.size() != n || !(margin >= 0.0)) {
    throw std::invalid_argument("class_agreement_with: another graph's solution or a bad margin");
  }
  class_agreement out;
  for (std::size_t v = 0; v < n; ++v) {
    if (present[v] == 0 || seed_labels[v] >= 0 || f.determined[v] == 0) {
      continue;
    }
    ++out.compared;
    if ((labels[v] >= 0.5F) == (f.solution[v] >= 0.5)) {
      continue;
    }
    ++out.flips;
    const double distance = std::fabs(f.solution[v] - 0.5);
    out.largest_flip_distance = std::max(out.largest_flip_distance, distance);
    if (distance > margin) {
      ++out.flips_beyond_margin;
    }
  }
  return out;
}

template harmonic_result harmonic_solution<std::int32_t>(
    const std::vector<std::int32_t>&, const std::vector<std::int32_t>&, const std::vector<float>&,
    const std::vector<std::uint8_t>&, const std::vector<std::int8_t>&, const harmonic_options&);
template harmonic_result harmonic_solution<std::int64_t>(
    const std::vector<std::int64_t>&, const std::vector<std::int32_t>&, const std::vector<float>&,
    const std::vector<std::uint8_t>&, const std::vector<std::int8_t>&, const harmonic_options&);

}  // namespace dyng::testing
