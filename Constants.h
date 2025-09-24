
#pragma once

// Any necessary includes would go here
#include <cstdint>  
#include <limits>  
// Use std::int32_t now

//use new instead
/**
 * @file Constants.h
 * @brief Physical and mathematical constants for ARBD simulations
 * @details Contains all constant values used throughout the ARBD project
 */

namespace ARBD {
    // Main namespace for Atomic Resolution Brownian Dynamics
    namespace constants {
    constexpr float PI = 3.141592653589793f;
    constexpr float TWOPI = 2.0f * PI;
    constexpr float HALFPI = 0.5f * PI;
    
    // Physical constants
    constexpr float COULOMB = 332.0636f;
    constexpr float BOLTZMANN = 0.001987191f;
    constexpr float TIMEFACTOR = 48.88821f;
    constexpr float PRESSUREFACTOR = 6.95E4f;
    
    // Simulation constants
    constexpr float PDBVELFACTOR = 20.45482706f;
    constexpr float PDBVELINVFACTOR = 1.0f/PDBVELFACTOR;
    constexpr float PNPERKCALMOL = 69.479f;
    constexpr float SMALLRAD = 0.0005f;
    constexpr float SMALLRAD2 = SMALLRAD * SMALLRAD;
    }

    // Use inline variables for configuration constants
    inline constexpr int MAX_NEIGHBORS = 27;

}

