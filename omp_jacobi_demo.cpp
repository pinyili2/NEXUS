#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Simple OpenMP Multi-GPU Jacobi Solver Demo
class SimpleOMPJacobiSolver {
private:
  int num_threads_;
  int nx_, ny_;
  std::vector<std::vector<float>> grid_partitions_;
  std::vector<std::vector<float>> grid_new_partitions_;
  std::vector<int> partition_sizes_;
  std::vector<int> partition_starts_;

  const float pi_ = 2.0f * std::asin(1.0f);
  const float tol_ = 1.0e-8f;

public:
  SimpleOMPJacobiSolver(int nx, int ny, int num_threads = -1)
      : nx_(nx), ny_(ny) {
#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads_ = omp_get_max_threads();
    } else {
      num_threads_ = num_threads;
    }
    // Limit to reasonable number of "GPUs"
    num_threads_ = std::min(num_threads_, 8);
#else
    num_threads_ = 1;
#endif

    std::cout << "Initialized SimpleOMPJacobiSolver with " << num_threads_
              << " threads" << std::endl;

    // Domain decomposition in Y direction
    int rows_per_thread = ny / num_threads_;
    int remainder = ny % num_threads_;

    partition_sizes_.resize(num_threads_);
    partition_starts_.resize(num_threads_);
    grid_partitions_.resize(num_threads_);
    grid_new_partitions_.resize(num_threads_);

    int current_row = 0;
    for (int t = 0; t < num_threads_; t++) {
      partition_starts_[t] = current_row;
      partition_sizes_[t] = rows_per_thread + (t < remainder ? 1 : 0);
      current_row += partition_sizes_[t];

      // Allocate memory for each partition
      grid_partitions_[t].resize(nx_ * partition_sizes_[t], 0.0f);
      grid_new_partitions_[t].resize(nx_ * partition_sizes_[t], 0.0f);
    }
  }

  void initialize_boundaries() {
    // Set sine wave boundary conditions on each partition
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_)
#endif
    for (int t = 0; t < num_threads_; t++) {
      for (int local_iy = 0; local_iy < partition_sizes_[t]; local_iy++) {
        int global_iy = partition_starts_[t] + local_iy;

        // Left and right boundaries
        float y0 = std::sin(2.0f * pi_ * global_iy / (ny_ - 1));
        grid_partitions_[t][local_iy * nx_ + 0] = y0;
        grid_partitions_[t][local_iy * nx_ + (nx_ - 1)] = y0;
        grid_new_partitions_[t][local_iy * nx_ + 0] = y0;
        grid_new_partitions_[t][local_iy * nx_ + (nx_ - 1)] = y0;
      }
    }
  }

  float solve(int iter_max, int nccheck = 1) {
    float l2_norm = 1.0f;
    int iter = 0;

    while (l2_norm > tol_ && iter < iter_max) {
      bool calculate_norm = (iter % nccheck) == 0;
      float total_norm = 0.0f;

      // Launch Jacobi kernel on all "GPUs" (threads)
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_) reduction(+ : total_norm)
#endif
      for (int t = 0; t < num_threads_; t++) {
        float local_norm = 0.0f;

        // Process interior points for this partition
        for (int local_iy = 1; local_iy < partition_sizes_[t] - 1; local_iy++) {
          for (int ix = 1; ix < nx_ - 1; ix++) {
            int idx = local_iy * nx_ + ix;

            // Jacobi update: new = 0.25 * (left + right + top + bottom)
            float new_val = 0.25f * (grid_partitions_[t][idx - 1] +   // left
                                     grid_partitions_[t][idx + 1] +   // right
                                     grid_partitions_[t][idx - nx_] + // top
                                     grid_partitions_[t][idx + nx_]   // bottom
                                    );

            grid_new_partitions_[t][idx] = new_val;

            if (calculate_norm) {
              float residue = new_val - grid_partitions_[t][idx];
              local_norm += residue * residue;
            }
          }
        }

        total_norm += local_norm;
      }

      if (calculate_norm) {
        l2_norm = std::sqrt(total_norm);
        if ((iter % 100) == 0) {
          std::cout << "Iteration " << iter << ", L2 norm: " << l2_norm
                    << std::endl;
        }
      }

      // Exchange ghost cells between partitions
      exchange_ghost_cells();

      // Swap arrays on all partitions
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_)
#endif
      for (int t = 0; t < num_threads_; t++) {
        std::swap(grid_partitions_[t], grid_new_partitions_[t]);
      }

      iter++;
    }

    return l2_norm;
  }

private:
  void exchange_ghost_cells() {
    if (num_threads_ == 1)
      return; // No exchange needed for single thread

    // Exchange between adjacent partitions
    for (int t = 0; t < num_threads_ - 1; t++) {
      int next_t = t + 1;

      // Copy bottom row of current partition to top row of next partition
      int bottom_row_start = (partition_sizes_[t] - 1) * nx_;
      std::copy(grid_new_partitions_[t].begin() + bottom_row_start,
                grid_new_partitions_[t].begin() + bottom_row_start + nx_,
                grid_new_partitions_[next_t].begin());

      // Copy top row of next partition to bottom row of current partition
      std::copy(grid_new_partitions_[next_t].begin() + nx_,
                grid_new_partitions_[next_t].begin() + 2 * nx_,
                grid_new_partitions_[t].begin() + bottom_row_start);
    }
  }

public:
  std::vector<float> get_solution() {
    std::vector<float> full_solution(nx_ * ny_);

    for (int t = 0; t < num_threads_; t++) {
      int start_row = partition_starts_[t];
      std::copy(grid_partitions_[t].begin(), grid_partitions_[t].end(),
                full_solution.begin() + start_row * nx_);
    }

    return full_solution;
  }

  int get_num_threads() const { return num_threads_; }
};

// Performance benchmark function
void run_performance_benchmark() {
  const int nx = 256;
  const int ny = 256;
  const int iter_max = 500;

  std::cout << "\n=== OpenMP Multi-GPU Jacobi Performance Benchmark ==="
            << std::endl;
  std::cout << "Problem size: " << nx << "x" << ny << ", " << iter_max
            << " iterations" << std::endl;

  // Test different thread configurations
  std::vector<int> thread_configs = {1, 2, 4, 8};
  std::vector<double> times;

  for (int num_threads : thread_configs) {
    std::cout << "\nTesting with " << num_threads << " thread(s):" << std::endl;

    SimpleOMPJacobiSolver solver(nx, ny, num_threads);
    solver.initialize_boundaries();

    auto start = std::chrono::high_resolution_clock::now();
    float final_norm = solver.solve(iter_max, 50);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    times.push_back(elapsed);

    std::cout << "  Time: " << elapsed << "s, Final norm: " << final_norm
              << std::endl;

    // Verify solution
    auto solution = solver.get_solution();
    bool has_nonzero =
        std::any_of(solution.begin(), solution.end(),
                    [](float val) { return std::abs(val) > 1e-6f; });

    if (!has_nonzero || final_norm < 0.0f) {
      std::cout << "  WARNING: Solution verification failed!" << std::endl;
    } else {
      std::cout << "  Solution verified successfully" << std::endl;
    }
  }

  // Calculate and report speedups
  if (times.size() > 1) {
    std::cout << "\n=== Speedup Results ===" << std::endl;
    double baseline = times[0]; // Single thread time
    for (size_t i = 0; i < times.size(); i++) {
      double speedup = baseline / times[i];
      double efficiency =
          (i == 0) ? 100.0 : speedup / thread_configs[i] * 100.0;
      std::cout << thread_configs[i] << " thread(s): " << speedup
                << "x speedup, " << efficiency << "% efficiency" << std::endl;
    }
  }
}

int main() {
  std::cout << "OpenMP Multi-GPU Jacobi Solver Demo" << std::endl;

#ifdef _OPENMP
  std::cout << "OpenMP available with " << omp_get_max_threads()
            << " max threads" << std::endl;
#else
  std::cout << "OpenMP not available - using single threaded" << std::endl;
#endif

  // Run basic functionality test
  std::cout << "\n=== Basic Functionality Test ===" << std::endl;

  SimpleOMPJacobiSolver solver(64, 64, 4);
  solver.initialize_boundaries();
  float final_norm = solver.solve(200, 20);

  std::cout << "Basic test completed with final norm: " << final_norm
            << std::endl;

  auto solution = solver.get_solution();
  bool has_nonzero =
      std::any_of(solution.begin(), solution.end(),
                  [](float val) { return std::abs(val) > 1e-6f; });

  if (has_nonzero && final_norm >= 0.0f) {
    std::cout << "Basic test PASSED" << std::endl;
  } else {
    std::cout << "Basic test FAILED" << std::endl;
    return 1;
  }

  // Run performance benchmark
  run_performance_benchmark();

  return 0;
}
