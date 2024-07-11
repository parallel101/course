#include <optional>
#include <print>
#include <chrono>
#include <string>

using namespace std::chrono_literals;

struct Connection {
    int fd;

    explicit Connection(int fd_) : fd(fd_) {
    }

    Connection &read();
};

struct ConnectionBuilderBase {
    std::string serverAddress;
    int port;
    bool useSSH = false;
    std::string sshCertPath = "";
    std::string sshPKeyPath = "";
    std::string sshCAFilePath = "";
    std::string username = "admin";
    std::string password = "password";
    bool enableFastTCPOpen = true;
    int tlsVersion = 1;
    std::chrono::seconds connectTimeout = 10s;
    std::chrono::seconds readTimeout = 5s;
    std::vector<std::string> args;
};

template <bool Ready = false>
struct [[nodiscard]] ConnectionBuilder : ConnectionBuilderBase {
    [[nodiscard]] ConnectionBuilder<true> &withAddress(std::string addr) {
        serverAddress = addr;
        return static_cast<ConnectionBuilder<true> &>(static_cast<ConnectionBuilderBase &>(*this));
    }

    [[nodiscard]] ConnectionBuilder &withPort(int p) {
        port = p;
        return *this;
    }

    [[nodiscard]] ConnectionBuilder<true> &withAddressAndPort(std::string addr) {
        auto pos = addr.find(':');
        serverAddress = addr.substr(0, pos);
        port = std::stoi(addr.substr(pos + 1));
        return static_cast<ConnectionBuilder<true> &>(static_cast<ConnectionBuilderBase &>(*this));
    }

    [[nodiscard]] ConnectionBuilder &withSSH(std::string cert, std::string pkey, std::string caf = "asas") {
        useSSH = true;
        sshCertPath = cert;
        sshPKeyPath = pkey;
        sshCAFilePath = caf;
        return *this;
    }

    [[nodiscard]] ConnectionBuilder &addArg(std::string arg) {
        args.push_back(arg);
        return *this;
    }

    [[nodiscard]] Connection connect() {
        static_assert(Ready, "你必须指定 addr 参数！");
        int fd = 0;
        // fd = open(serverAddress, port);
        return Connection(fd);
    }
};

Connection c = ConnectionBuilder()
    .withSSH("1", "2")
    .addArg("asas")
    .addArg("bsbs")
    .withAddressAndPort("localhost:8080")
    .addArg("baba")
    .connect();

struct [[nodiscard]] Cake {
    int handle;

    Cake() {}

    [[nodiscard]] Cake &&setOrig() && {
        // 构造原味蛋糕
        handle = 0;
        return std::move(*this);
    }

    [[nodiscard]] Cake &&setChoco(double range) && {
        // 构造巧克力蛋糕
        handle = (int)range;
        return std::move(*this);
    }

    [[nodiscard]] Cake &&setMoca(int flavor) && {
        // 构造抹茶味蛋糕
        handle = flavor;
        return std::move(*this);
    }

    Cake(Cake &&) = default;
    Cake(Cake const &) = delete;
};

void func(Cake &&c);
void func(Cake const &c);

Cake origCake = Cake().setOrig().setChoco(1.0);
Cake chocoCake = Cake().setChoco(1.0);
Cake matchaCake = Cake().setMoca(1);

int main() {
    Cake c;
    std::move(c).setOrig();
    Cake().setOrig();
    func(std::move(c));
}
