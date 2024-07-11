#include <optional>
#include <print>
#include <chrono>
#include <string>

using namespace std::chrono_literals;

struct Connection {
    int fd;

    explicit Connection(int fd_) : fd(fd_) {
    }
};

struct ConnectionBuilder {
    std::string serverAddress;
    int port;
    struct SSHParams {
        std::string sshCertPath = "";
        std::string sshPKeyPath = "";
        std::string sshCAFilePath = "";
    };
    std::optional<SSHParams> useSSH;
    std::string username = "admin";
    std::string password = "password";
    bool enableFastTCPOpen = true;
    int tlsVersion = 1;
    std::chrono::seconds connectTimeout = 10s;
    std::chrono::seconds readTimeout = 5s;

    Connection connect() {
        int fd = 0;
        // fd = open(serverAddress, port);
        return Connection(fd);
    }
};

Connection c = ConnectionBuilder{
             .serverAddress = "localhost",
             .port = 8080,
             .useSSH = std::nullopt,
         }.connect();

struct Cake {
    int handle;

    explicit Cake(int han) : handle(han) {}

    static Cake makeOrig() {
        // 构造原味蛋糕
        int han = 0;
        return Cake(han);
    }

    static Cake makeChoco(double range) {
        // 构造巧克力蛋糕
        int han = (int)range;
        return Cake(han);
    }

    static Cake makeMoca(int flavor) {
        // 构造抹茶味蛋糕
        int han = flavor;
        return Cake(han);
    }
};

Cake origCake = Cake::makeOrig();
Cake chocoCake = Cake::makeChoco(1.0);
Cake matchaCake = Cake::makeMoca(1);
