#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <vector>

// Raw MPI test without MPIManager singleton
TEST_CASE("Raw MPI Test", "[mpi][raw]") {
#ifdef USE_MPI
  // Initialize MPI directly
  int rank, size;
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  INFO("Running on rank " << rank << " of " << size << " processes");

  SECTION("Basic MPI Communication") {
    // Test basic MPI communication
    int send_data = rank * 10;
    int recv_data = 0;

    MPI_Allreduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int expected_sum = 0;
    for (int i = 0; i < size; i++) {
      expected_sum += i * 10;
    }

    REQUIRE(recv_data == expected_sum);
    INFO("Rank " << rank
                 << ": MPI communication successful, sum = " << recv_data);
  }

  SECTION("Point-to-Point Communication") {
    if (size >= 2) {
      int send_data = rank * 100;
      int recv_data = 0;

      if (rank == 0) {
        MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        REQUIRE(recv_data == 100);
        INFO("Rank 0: Point-to-point communication successful");
      } else if (rank == 1) {
        MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&send_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        REQUIRE(recv_data == 0);
        INFO("Rank 1: Point-to-point communication successful");
      }
    } else {
      INFO("Skipping point-to-point test - need at least 2 processes");
    }
  }

  SECTION("Broadcast Test") {
    int data = 0;
    if (rank == 0) {
      data = 42;
    }

    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    REQUIRE(data == 42);
    INFO("Rank " << rank << ": Broadcast test successful, received " << data);
  }

  SECTION("Scatter and Gather Test") {
    std::vector<int> send_data;
    std::vector<int> recv_data(1);

    if (rank == 0) {
      send_data.resize(size);
      for (int i = 0; i < size; i++) {
        send_data[i] = i * 10;
      }
    }

    MPI_Scatter(send_data.data(), 1, MPI_INT, recv_data.data(), 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    REQUIRE(recv_data[0] == rank * 10);
    INFO("Rank " << rank << ": Scatter test successful, received "
                 << recv_data[0]);

    // Gather back
    std::vector<int> gather_data;
    if (rank == 0) {
      gather_data.resize(size);
    }

    MPI_Gather(recv_data.data(), 1, MPI_INT, gather_data.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
      for (int i = 0; i < size; i++) {
        REQUIRE(gather_data[i] == i * 10);
      }
      INFO("Rank 0: Gather test successful");
    }
  }

  SECTION("6 Process Communication") {
    // Each process sends its rank * 10
    int send_data = rank * 10;
    int recv_data = 0;

    // All-to-all communication
    MPI_Allreduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int expected_sum = 0;
    for (int i = 0; i < size; i++) {
      expected_sum += i * 10;
    }

    REQUIRE(recv_data == expected_sum);
    INFO("Rank " << rank << ": 6-process communication successful, sum = "
                 << recv_data);
  }

  // Finalize MPI
  MPI_Finalize();

#else
  WARN("MPI not available - skipping MPI tests");
  REQUIRE(true);
#endif
}
