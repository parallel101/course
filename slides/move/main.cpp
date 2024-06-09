#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

struct IndentGuard {
    IndentGuard(std::string &indent_) : indent(indent_) {
        oldIndent = indent;
        indent += "  ";
    }

    IndentGuard(IndentGuard &&) = delete;

    ~IndentGuard() {
        indent = oldIndent;
    }

    std::string oldIndent;
    std::string &indent;
};

struct Codegen {
    std::string code;
    std::string indent;

    void emit(std::string text) {
        code += indent + text + "\n";
    }

    void emit_variable(std::string name) {
        code += indent + "int " + name + ";\n";
    }

    void codegen() {
        emit("int main() {");
        {
            IndentGuard guard(indent);
            emit_variable("i");
            emit_variable("j");
        }
        emit("}");
        emit_variable("g");
    }
};

int main() {
    Codegen cg;
    cg.codegen();
    std::cout << cg.code;
    return 0;
}
