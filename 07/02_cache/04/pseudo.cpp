// fully SOA, **bad**, for small size simd use
struct ParticleList {
    std::vector<float> pos_x;
    std::vector<float> pos_y;
    std::vector<float> pos_z;
    std::vector<float> vel_x;
    std::vector<float> vel_y;
    std::vector<float> vel_z;
};

// partially SOA, partially AOS, **good**, for daily use
struct ParticleList {
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> vel;
};

// littlemine's favo AOSOA, **good**, for hpc experts
struct ParticleBlock {
    float pos_x[1024];
    float pos_y[1024];
    float pos_z[1024];
    float vel_x[1024];
    float vel_y[1024];
    float vel_z[1024];
};
using ParticleList = std::vector<ParticleBlock>;

// fully AOS, **bad**, for naive students
struct Particle {
    glm::vec3 pos;
    glm::vec3 vel;
};
using ParticleList = std::vector<Particle>;



