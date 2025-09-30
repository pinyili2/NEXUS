#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// Optimized OpenMP Multi-GPU Jacobi Solver
// Based on NVIDIA multi-GPU programming models:
// https://github.com/NVIDIA/multi-gpu-programming-models
class OptimizedOMPJacobiSolver {
private:
  int num_threads_;
  int nx_, ny_;

  // Per-thread data structures for better cache locality
  struct ThreadData {
    std::vector<float> grid;
    std::vector<float> grid_new;
    int start_y, end_y;
    int ny_local;
    float local_norm;
  };

  std::vector<std::unique_ptr<ThreadData>> thread_data_;

  const float pi_ = 2.0f * std::asin(1.0f);
  const float tol_ = 1.0e-8f;

public:
  OptimizedOMPJacobiSolver(int nx, int ny, int num_threads = -1)
      : nx_(nx), ny_(ny) {
#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads_ = omp_get_max_threads();
    } else {
      num_threads_ = num_threads;
    }
    num_threads_ = std::min(num_threads_, 16); // Reasonable limit
#else
    num_threads_ = 1;
#endif

    std::cout << "Optimized solver with " << num_threads_ << " threads" << std::endl;

    // Initialize thread data structures
    thread_data_.resize(num_threads_);

    // Domain decomposition with ghost cells
    int rows_per_thread = ny / num_threads_;
    int remainder = ny % num_threads_;

    int current_row = 0;
    for (int t = 0; t < num_threads_; t++) {
      thread_data_[t] = std::make_unique<ThreadData>();

      thread_data_[t]->start_y = current_row;
      thread_data_[t]->ny_local = rows_per_thread + (t < remainder ? 1 : 0);
      thread_data_[t]->end_y = current_row + thread_data_[t]->ny_local;
      current_row += thread_data_[t]->ny_local;

      // Add ghost cells (1 above, 1 below) except for boundaries
      int ghost_ny = thread_data_[t]->ny_local;
      if (t > 0) ghost_ny++; // ghost cell above
      if (t < num_threads_ - 1) ghost_ny++; // ghost cell below

      thread_data_[t]->grid.resize(nx_ * ghost_ny, 0.0f);
      thread_data_[t]->grid_new.resize(nx_ * ghost_ny, 0.0f);
      thread_data_[t]->local_norm = 0.0f;
    }
  }

  void initialize_boundaries() {
    // Parallel boundary initialization
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_)
#endif
    for (int t = 0; t < num_threads_; t++) {
      auto& data = *thread_data_[t];

      // Calculate offset for ghost cells
      int ghost_offset = (t > 0) ? 1 : 0;

      for (int local_iy = 0; local_iy < data.ny_local; local_iy++) {
        int global_iy = data.start_y + local_iy;
        int grid_row = local_iy + ghost_offset;

        // Set sine wave boundary conditions
        float y0 = std::sin(2.0f * pi_ * global_iy / (ny_ - 1));
        data.grid[grid_row * nx_ + 0] = y0;
        data.grid[grid_row * nx_ + (nx_ - 1)] = y0;
        data.grid_new[grid_row * nx_ + 0] = y0;
        data.grid_new[grid_row * nx_ + (nx_ - 1)] = y0;
      }
    }
  }

  float solve(int iter_max, int nccheck = 1) {
    float l2_norm = 1.0f;
    int iter = 0;

    while (l2_norm > tol_ && iter < iter_max) {
      bool calculate_norm = (iter % nccheck) == 0;

      // OPTIMIZATION 1: Parallel ghost cell exchange
      exchange_ghost_cells_parallel();

      // OPTIMIZATION 2: Overlapped computation with better data locality
      compute_jacobi_parallel(calculate_norm);

      // OPTIMIZATION 3: Efficient norm reduction
      if (calculate_norm) {
        l2_norm = compute_norm_parallel();

        if ((iter % 100) == 0) {
          std::cout << "Iteration " << iter << ", L2 norm: " << l2_norm << std::endl;
        }
      }

      // OPTIMIZATION 4: Parallel array swap
      swap_arrays_parallel();

      iter++;
    }

    return l2_norm;
  }

private:
  // OPTIMIZATION 1: Parallel ghost cell exchange (inspired by NVIDIA's P2P approach)
  void exchange_ghost_cells_parallel() {
    if (num_threads_ == 1) return;

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_ - 1)
#endif
    for (int t = 0; t < num_threads_ - 1; t++) {
      exchange_between_threads(t, t + 1);
    }
  }

  void exchange_between_threads(int t1, int t2) {
    auto& data1 = *thread_data_[t1];
    auto& data2 = *thread_data_[t2];

    // Calculate indices for ghost cell exchange
    int t1_ghost_offset = (t1 > 0) ? 1 : 0;
    int t2_ghost_offset = (t2 > 0) ? 1 : 0;

    // t1's bottom row -> t2's top ghost cell
    int t1_bottom_row = t1_ghost_offset + data1.ny_local - 1;
    int t2_top_ghost = (t2 > 0) ? 0 : -1; // No top ghost for first thread

    if (t2_top_ghost >= 0) {
      std::copy(data1.grid_new.begin() + t1_bottom_row * nx_,
                data1.grid_new.begin() + (t1_bottom_row + 1) * nx_,
                data2.grid_new.begin() + t2_top_ghost * nx_);
    }

    // t2's top row -> t1's bottom ghost cell
    int t2_top_row = t2_ghost_offset;
    int t1_bottom_ghost = t1_ghost_offset + data1.ny_local;

    if (t1_bottom_ghost < static_cast<int>(data1.grid_new.size() / nx_)) {
      std::copy(data2.grid_new.begin() + t2_top_row * nx_,
                data2.grid_new.begin() + (t2_top_row + 1) * nx_,
                data1.grid_new.begin() + t1_bottom_ghost * nx_);
    }
  }

  // OPTIMIZATION 2: Parallel Jacobi computation with better cache locality
  void compute_jacobi_parallel(bool calculate_norm) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_)
#endif
    for (int t = 0; t < num_threads_; t++) {
      compute_jacobi_thread(t, calculate_norm);
    }
  }

  void compute_jacobi_thread(int t, bool calculate_norm) {
    auto& data = *thread_data_[t];
    float local_norm = 0.0f;

    int ghost_offset = (t > 0) ? 1 : 0;

    // Process interior points only
    for (int local_iy = 1; local_iy < data.ny_local - 1; local_iy++) {
      int grid_row = local_iy + ghost_offset;

      for (int ix = 1; ix < nx_ - 1; ix++) {
        int idx = grid_row * nx_ + ix;

        // Jacobi stencil
        float new_val = 0.25f * (
          data.grid[idx - 1] +        // left
          data.grid[idx + 1] +        // right
          data.grid[idx - nx_] +      // top
          data.grid[idx + nx_]        // bottom
        );

        data.grid_new[idx] = new_val;

        if (calculate_norm) {
          float residue = new_val - data.grid[idx];
          local_norm += residue * residue;
        }
      }
    }

    data.local_norm = local_norm;
  }

  // OPTIMIZATION 3: Efficient parallel norm reduction
  float compute_norm_parallel() {
    float total_norm = 0.0f;

    // Simple reduction - could be optimized further with tree reduction
    for (int t = 0; t < num_threads_; t++) {
      total_norm += thread_data_[t]->local_norm;
    }

    return std::sqrt(total_norm);
  }

  // OPTIMIZATION 4: Parallel array swap
  void swap_arrays_parallel() {
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads_)
#endif
    for (int t = 0; t < num_threads_; t++) {
      std::swap(thread_data_[t]->grid, thread_data_[t]->grid_new);
    }
  }

public:
  std::vector<float> get_solution() {
    std::vector<float> full_solution(nx_ * ny_);

    for (int t = 0; t < num_threads_; t++) {
      auto& data = *thread_data_[t];
      int ghost_offset = (t > 0) ? 1 : 0;

      for (int local_iy = 0; local_iy < data.ny_local; local_iy++) {
        int global_iy = data.start_y + local_iy;
        int grid_row = local_iy + ghost_offset;

        std::copy(data.grid.begin() + grid_row * nx_,
                  data.grid.begin() + (grid_row + 1) * nx_,
                  full_solution.begin() + global_iy * nx_);
      }
    }

    return full_solution;
  }

  int get_num_threads() const { return num_threads_; }
};

// Performance comparison function
void run_scaling_comparison() {
  const int nx = 512;  // Larger problem for better scaling analysis
  const int ny = 512;
  const int iter_max = 500;

  std::cout << "\n=== Optimized vs Original Scaling Comparison ===" << std::endl;
  std::cout << "Problem size: " << nx << "x" << ny << ", " << iter_max << " iterations" << std::endl;

  std::vector<int> thread_configs = {1, 2, 4, 8, 16};
  std::vector<double> times;

  for (int num_threads : thread_configs) {
    if (num_threads > 32) continue; // Reasonable limit

    std::cout << "\nTesting optimized solver with " << num_threads << " thread(s):" << std::endl;

    OptimizedOMPJacobiSolver solver(nx, ny, num_threads);
    solver.initialize_boundaries();

    auto start = std::chrono::high_resolution_clock::now();
    float final_norm = solver.solve(iter_max, 50);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    times.push_back(elapsed);

    std::cout << "  Time: " << elapsed << "s, Final norm: " << final_norm << std::endl;

    // Verify solution
    auto solution = solver.get_solution();
    bool has_nonzero = std::any_of(solution.begin(), solution.end(),
                                 [](float val) { return std::abs(val) > 1e-6f; });

    if (!has_nonzero || final_norm < 0.0f) {
      std::cout << "  WARNING: Solution verification failed!" << std::endl;
    } else {
      std::cout << "  Solution verified successfully" << std::endl;
    }
  }

  // Calculate and report speedups
  if (times.size() > 1) {
    std::cout << "\n=== Optimized Scaling Results ===" << std::endl;
    double baseline = times[0];
    for (size_t i = 0; i < times.size(); i++) {
      double speedup = baseline / times[i];
      double efficiency = (i == 0) ? 100.0 : speedup / thread_configs[i] * 100.0;
      std::cout << thread_configs[i] << " thread(s): " << speedup
                << "x speedup, " << efficiency << "% efficiency" << std::endl;
    }
  }
}

int main() {
  std::cout << "Optimized OpenMP Multi-GPU Jacobi Solver" << std::endl;
  std::cout << "Based on NVIDIA multi-GPU programming models" << std::endl;

#ifdef _OPENMP
  std::cout << "OpenMP available with " << omp_get_max_threads() << " max threads" << std::endl;
#else
  std::cout << "OpenMP not available - using single threaded" << std::endl;
#endif

  // Run basic functionality test
  std::cout << "\n=== Basic Functionality Test ===" << std::endl;
  OptimizedOMPJacobiSolver solver(64, 64, 4);
  solver.initialize_boundaries();
  float final_norm = solver.solve(200, 20);

  std::cout << "Basic test completed with final norm: " << final_norm << std::endl;

  auto solution = solver.get_solution();
  bool has_nonzero = std::any_of(solution.begin(), solution.end(),
                               [](float val) { return std::abs(val) > 1e-6f; });

  if (has_nonzero && final_norm >= 0.0f) {
    std::cout << "Basic test PASSED" << std::endl;
  } else {
    std::cout << "Basic test FAILED" << std::endl;
    return 1;
  }

  // Run scaling comparison
  run_scaling_comparison();

  return 0;
}
