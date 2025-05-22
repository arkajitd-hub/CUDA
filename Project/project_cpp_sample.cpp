#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

#define NUM_NEUTRONS 1000000  // Number of neutrons to simulate
#define DOMAIN_SIZE 10.0      // Material size in cm
#define SIGMA_T 1.0           // Total macroscopic cross-section (1/cm)
#define SIGMA_A 0.2           // Absorption cross-section (1/cm)
#define SIGMA_F 0.1           // Fission cross-section (1/cm)

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist(0.0, 1.0);

// --- Structs for Data Representation --- //

// Struct to represent a neutron particle
struct Particle {
    double x, ux;       // Position and direction (1D for simplicity)
    double weight;      // Importance weight (for variance reduction)
    bool alive;         // If the particle is still active

    Particle(double x0) : x(x0), ux(1.0), weight(1.0), alive(true) {}
};

// Struct to represent a material in the simulation
struct Material {
    double sigma_t;  // Total macroscopic cross-section
    double sigma_a;  // Absorption cross-section
    double sigma_f;  // Fission cross-section

    Material(double st, double sa, double sf) : sigma_t(st), sigma_a(sa), sigma_f(sf) {}
};

// Struct to store flux and reaction rates
struct Tally {
    double flux;
    double absorption_rate;
    double fission_rate;
    double keff;
    
    Tally() : flux(0), absorption_rate(0), fission_rate(0), keff(0) {}
};

// --- Function to Sample Distance to Next Collision ---
double sample_distance(double sigma_t) {
    return -std::log(dist(gen)) / sigma_t;
}

// --- Function to Simulate a Neutron Transport ---
void simulate_neutron(Particle &neutron, const Material &material, Tally &tally) {
    while (neutron.alive) {
        // Sample distance to next collision
        double step = sample_distance(material.sigma_t);
        neutron.x += step;

        // Check if neutron has left the domain
        if (neutron.x >= DOMAIN_SIZE) {
            neutron.alive = false;  // Kill neutron if it leaves the material
            return;
        }

        // Sample reaction type (absorption, fission, scattering)
        double xi = dist(gen);
        if (xi < (material.sigma_a / material.sigma_t)) {
            // Absorption event
            tally.absorption_rate += neutron.weight;
            neutron.alive = false;
        } else if (xi < ((material.sigma_a + material.sigma_f) / material.sigma_t)) {
            // Fission event (assume 2 new neutrons per fission)
            tally.fission_rate += neutron.weight;
            tally.keff += 2.0;  // Assume 2 neutrons produced per fission
            neutron.alive = false;
        } else {
            // Scattering event (randomize direction)
            neutron.ux = (dist(gen) < 0.5) ? -1.0 : 1.0;
        }

        // Update neutron flux tally (path length estimator)
        tally.flux += neutron.weight * step;
    }
}

// --- Main Function to Run Monte Carlo Simulation ---
int main() {
    Material fuel(SIGMA_T, SIGMA_A, SIGMA_F);  // Define material properties
    Tally tally;  // Initialize tally

    std::cout << "Simulating " << NUM_NEUTRONS << " neutrons...\n";

    // OpenMP Parallelization for Multi-Core Performance
    #pragma omp parallel for reduction(+:tally.flux, tally.absorption_rate, tally.fission_rate, tally.keff)
    for (int i = 0; i < NUM_NEUTRONS; i++) {
        Particle neutron(0.0);  // Start neutron at x = 0
        simulate_neutron(neutron, fuel, tally);
    }

    // Normalize Flux
    tally.flux /= NUM_NEUTRONS;

    // Compute Final k-effective
    tally.keff /= NUM_NEUTRONS;

    // Print Results
    std::cout << "Total Flux: " << tally.flux << " neutrons/cm^2\n";
    std::cout << "Absorption Rate: " << tally.absorption_rate / NUM_NEUTRONS << " reactions/neutron\n";
    std::cout << "Fission Rate: " << tally.fission_rate / NUM_NEUTRONS << " reactions/neutron\n";
    std::cout << "Final k-effective: " << tally.keff << "\n";

    return 0;
}
