#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <termios.h>

struct StdinRawify {
    struct termios oldtc;
    bool saved;

    StdinRawify() {
        saved = false;
        if (isatty(STDIN_FILENO)) {
            struct termios tc;
            tcgetattr(STDIN_FILENO, &tc);
            memcpy(&oldtc, &tc, sizeof(tc));
            saved = true;
            tc.c_lflag &= ~ICANON;
            tc.c_lflag &= ~ECHO;
            tcsetattr(STDIN_FILENO, TCSANOW, &tc);
        }
    }

    ~StdinRawify() {
        if (saved) {
            tcsetattr(STDIN_FILENO, TCSANOW, &oldtc);
        }
    }
};

std::string input_password(const char *prompt, size_t max_size = static_cast<size_t>(-1)) {
    if (prompt) {
        fprintf(stderr, "%s", prompt);
    }
    std::string ret;
    StdinRawify stdinRawifier;
    while (true) {
        int c = getchar();
        if (c == EOF)
            break;
        if (c == '\n' || c == '\r') {
            fputc('\n', stderr);
            break;
        } else if (c == '\b' || c == '\x7f') {
            if (ret.size() > 0) {
                ret.pop_back();
                fprintf(stderr, "\b \b");
            }
        } else {
            if (ret.size() < max_size) {
                ret.push_back(c);
                fputc('*', stderr);
            }
        }
    }
    return ret;
}

int main() {
    auto passwd = input_password("请输入密码：");
    // setvbuf(stdin, nullptr, _IONBF, 0);
    fprintf(stderr, "输入的密码是：%s\n", passwd.c_str());
    return 0;
}
