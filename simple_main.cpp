#include "ARBDException.h"
#include "Backend/Resource.h"
#include "SignalManager.h"
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char *argv[]) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif

  std::cout << "MultiBack Library Test Application" << std::endl;
  std::cout << "Successfully linked with libMultiBack!" << std::endl;

#ifdef USE_CUDA
  std::cout << "CUDA backend enabled" << std::endl;
#endif

#ifdef USE_SYCL
  std::cout << "SYCL backend enabled" << std::endl;
#endif

#ifdef USE_METAL
  std::cout << "Metal backend enabled" << std::endl;
#endif

#ifdef USE_MPI
  std::cout << "MPI support enabled" << std::endl;
  MPI_Finalize();
#endif

  return 0;
}
