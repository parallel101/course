#include <cstdio>
#include <cstring>
#include <map>
#include <cerrno>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

using namespace std;

struct InStream {
    virtual size_t read(char *__restrict s, size_t len) = 0;
    virtual ~InStream() = default;

    virtual int getchar() {
        char c;
        size_t n = read(&c, 1);
        if (n == 0) {
            return EOF;
        }
        return c;
    }

    virtual size_t readn(char *__restrict s, size_t len) {
        size_t n = read(s, len);
        if (n == 0) return 0;
        while (n != len) {
            size_t m = read(s + n, len - n);
            if (m == 0) break;
            n += m;
        }
        return n;
    }

    std::string readall() {
        std::string ret;
        ret.resize(32);
        char *buf = &ret[0];
        size_t pos = 0;
        while (true) {
            size_t n = read(buf + pos, ret.size() - pos);
            if (n == 0) {
                break;
            }
            pos += n;
            if (pos == ret.size()) {
                ret.resize(ret.size() * 2);
            }
        }
        ret.resize(pos);
        return ret;
    }

    std::string readuntil(char eol) {
        std::string ret;
        while (true) {
            int c = getchar();
            if (c == EOF) {
                break;
            }
            ret.push_back(c);
            if (c == eol) {
                break;
            }
        }
        return ret;
    }

    std::string readuntil(const char *__restrict eol, size_t neol) {
        std::string ret;
        while (true) {
            int c = getchar();
            if (c == EOF) {
                break;
            }
            ret.push_back(c);
            if (ret.size() >= neol) {
                if (memcmp(ret.data() + ret.size() - neol, eol, neol) == 0) {
                    break;
                }
            }
        }
        return ret;
    }

    std::string readuntil(std::string const &eol) {
        return readuntil(eol.data(), eol.size());
    }

#if __cpp_lib_string_view
    void readline(std::string_view eol) {
        readline(eol.data(), eol.size());
    }
#endif

    std::string getline(char eol) {
        std::string ret = readuntil(eol);
        if (ret.size() > 0 && ret.back() == eol)
            ret.pop_back();
        return ret;
    }

    std::string getline(const char *__restrict eol, size_t neol) {
        std::string ret = readuntil(eol, neol);
        if (ret.size() >= neol && memcmp(ret.data() + ret.size() - neol, eol, neol) == 0)
            ret.resize(ret.size() - neol);
        return ret;
    }

    std::string getline(std::string const &eol) {
        return getline(eol.data(), eol.size());
    }

#if __cpp_lib_string_view
    void getline(std::string_view eol) {
        getline(eol.data(), eol.size());
    }
#endif
};

struct BufferedInStream : InStream {
private:
    InStream &in;
    char *buf;
    size_t top = 0;
    size_t max = 0;

    [[nodiscard]] bool refill() {
        top = 0;
        max = in.read(buf, BUFSIZ);
        // max <= BUFSIZ
        return max != 0;
    }

public:
    explicit BufferedInStream(InStream &in_)
        : in(in_)
    {
        buf = (char *)valloc(BUFSIZ);
    }

    int getchar() override {
        if (top == max) {
            if (!refill())
                return EOF;
        }
        return buf[top++];
    }

    size_t read(char *__restrict s, size_t len) override {
        // 如果缓冲区为空，则阻塞，否则尽量不阻塞，返回已缓冲的字符
        char *__restrict p = s;
        while (p != s + len) {
            if (top == max) {
                if (p != s || !refill())
                    break;
            }
            int c = buf[top++];
            *p++ = c;
        }
        return p - s;
    }

    size_t readn(char *__restrict s, size_t len) override {
        // 尽量读满 len 个字符，除非遇到 EOF，才会返回小于 len 的值
        char *__restrict p = s;
        while (p != s + len) {
            if (top == max) {
                if (!refill())
                    break;
            }
            int c = buf[top++];
            *p++ = c;
        }
        return p - s;
    }

    BufferedInStream(BufferedInStream &&) = delete;

    ~BufferedInStream() {
        free(buf);
    }

    InStream &base_stream() const noexcept {
        return in;
    }
};

struct BufferedInStreamOwned : BufferedInStream {
    explicit BufferedInStreamOwned(std::unique_ptr<InStream> in)
        : BufferedInStream(*in) {
        (void)in.release();
    }

    BufferedInStreamOwned(BufferedInStreamOwned &&) = delete;

    ~BufferedInStreamOwned() {
        delete &base_stream();
    }
};

struct UnixFileInStream : InStream {
private:
    int fd;

public:
    explicit UnixFileInStream(int fd_) : fd(fd_) {
    }

    size_t read(char *__restrict s, size_t len) override {
        if (len == 0) return 0;
        std::this_thread::sleep_for(0.2s);
        ssize_t n = ::read(fd, s, len);
        if (n < 0)
            throw std::system_error(errno, std::generic_category());
        return n;
    }

    int file_handle() const noexcept {
        return fd;
    }
};

struct UnixFileInStreamOwned : UnixFileInStream {
    explicit UnixFileInStreamOwned(int fd_) : UnixFileInStream(fd_) {
    }

    ~UnixFileInStreamOwned() {
        ::close(file_handle());
    }
};

struct OutStream {
    virtual void write(const char *__restrict s, size_t len) = 0;
    virtual ~OutStream() = default;

    void puts(const char *__restrict s) {
        write(s, strlen(s));
    }

    void puts(std::string const &s) {
        write(s.data(), s.size());
    }

#if __cpp_lib_string_view
    void puts(std::string_view s) {
        write(s.data(), s.size());
    }
#endif

    virtual void putchar(char c) {
        write(&c, 1);
    }

    virtual void flush() {
    }
};

struct UnixFileOutStream : OutStream {
private:
    int fd;

public:
    explicit UnixFileOutStream(int fd_) : fd(fd_) {
    }

    void write(const char *__restrict s, size_t len) override {
        if (len == 0) return;
        std::this_thread::sleep_for(0.2s);
        ssize_t written = ::write(fd, s, len);
        if (written < 0)
            throw std::system_error(errno, std::generic_category());
        if (written == 0)
            throw std::system_error(EPIPE, std::generic_category());
        while ((size_t)written != len) {
            written = ::write(fd, s, len);
            if (written < 0)
                throw std::system_error(errno, std::generic_category());
            if (written == 0)
                throw std::system_error(EPIPE, std::generic_category());
            s += written;
            len -= written;
        }
    }

    UnixFileOutStream(UnixFileOutStream &&) = delete;

    ~UnixFileOutStream() {
        ::close(fd);
    }

    int file_handle() const noexcept {
        return fd;
    }
};

struct UnixFileOutStreamOwned : UnixFileOutStream {
    explicit UnixFileOutStreamOwned(int fd_) : UnixFileOutStream(fd_) {
    }

    ~UnixFileOutStreamOwned() {
        ::close(file_handle());
    }
};

struct BufferedOutStream : OutStream {
private:
    OutStream &out;
    char *buf;
    size_t top = 0;

public:
    explicit BufferedOutStream(OutStream &out_)
        : out(out_)
        , buf((char *)valloc(BUFSIZ))
    {
    }

    void flush() override {
        if (top) {
            out.write(buf, top);
            top = 0;
        }
    }

    void putchar(char c) override {
        if (top == BUFSIZ)
            flush();
        buf[top++] = c;
    }

    void write(const char *__restrict s, size_t len) override {
        for (const char *__restrict p = s; p != s + len; ++p) {
            if (top == BUFSIZ)
                flush();
            char c = *p;
            buf[top++] = c;
        }
    }

    BufferedOutStream(BufferedOutStream &&) = delete;

    ~BufferedOutStream() {
        flush();
        free(buf);
    }

    OutStream &base_stream() const noexcept {
        return out;
    }
};

struct BufferedOutStreamOwned : BufferedOutStream {
    explicit BufferedOutStreamOwned(std::unique_ptr<OutStream> out)
        : BufferedOutStream(*out) {
        (void)out.release();
    }

    BufferedOutStreamOwned(BufferedOutStreamOwned &&) = delete;

    ~BufferedOutStreamOwned() {
        delete &base_stream();
    }
};

struct LineBufferedOutStream : OutStream {
private:
    OutStream &out;
    char *buf;
    size_t top = 0;

public:
    explicit LineBufferedOutStream(OutStream &out_)
        : out(out_)
        , buf((char *)valloc(BUFSIZ))
    {
    }

    void flush() override {
        if (top) {
            out.write(buf, top);
            top = 0;
        }
    }

    void putchar(char c) override {
        if (top == BUFSIZ)
            flush();
        buf[top++] = c;
        if (c == '\n')
            flush();
    }

    void write(const char *__restrict s, size_t len) override {
        for (const char *__restrict p = s; p != s + len; ++p) {
            if (top == BUFSIZ)
                flush();
            char c = *p;
            buf[top++] = c;
            if (c == '\n')
                flush();
        }
    }

    LineBufferedOutStream(BufferedOutStream &&) = delete;

    ~LineBufferedOutStream() {
        flush();
        free(buf);
    }

    OutStream &base_stream() const noexcept {
        return out;
    }
};

struct LineBufferedOutStreamOwned : LineBufferedOutStream {
    explicit LineBufferedOutStreamOwned(std::unique_ptr<OutStream> out)
        : LineBufferedOutStream(*out) {
        (void)out.release();
    }

    LineBufferedOutStreamOwned(BufferedOutStreamOwned &&) = delete;

    ~LineBufferedOutStreamOwned() {
        delete &base_stream();
    }
};

BufferedInStreamOwned io_in(std::make_unique<UnixFileInStream>(STDIN_FILENO));
LineBufferedOutStreamOwned io_out(std::make_unique<UnixFileOutStream>(STDOUT_FILENO));
UnixFileOutStream io_err(STDERR_FILENO);

void io_perror(const char *msg) {
    io_err.puts(msg);
    io_err.puts(": ");
    io_err.puts(strerror(errno));
    io_err.puts("\n");
}

enum OpenFlag {
    Read,
    Write,
    Append,
    ReadWrite,
};

std::map<OpenFlag, int> openFlagToUnixFlag = {
    {OpenFlag::Read, O_RDONLY},
    {OpenFlag::Write, O_WRONLY | O_TRUNC | O_CREAT},
    {OpenFlag::Append, O_WRONLY | O_APPEND | O_CREAT},
    {OpenFlag::ReadWrite, O_RDWR | O_CREAT},
};

std::unique_ptr<OutStream> out_file_open(const char *path, OpenFlag flag) {
    int oflag = openFlagToUnixFlag.at(flag);
    int fd = open(path, oflag);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category());
    }
    auto file = std::make_unique<UnixFileOutStream>(fd);
    return file;
    // return std::make_unique<BufferedOutStream>(std::move(file));
}

std::unique_ptr<InStream> in_file_open(const char *path, OpenFlag flag) {
    int oflag = openFlagToUnixFlag.at(flag);
    int fd = open(path, oflag);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category());
    }
    auto file = std::make_unique<UnixFileInStream>(fd);
    return file;
    // return std::make_unique<BufferedInStream>(std::move(file));
}

int main() {
    {
        auto p = out_file_open("/tmp/a.txt", OpenFlag::Write);
        p->puts("Hello!\nWorld!\n");
    }
    {
        auto p = in_file_open("/tmp/a.txt", OpenFlag::Read);
        auto s = p->getline('\n');
        printf("%s\n", s.c_str());
        s = p->getline('\n');
        printf("%s\n", s.c_str());
        s = p->getline('\n');
        printf("%s\n", s.c_str());
    }
}
