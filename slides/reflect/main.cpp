#include <cassert>
#include <string>

struct Student {
    std::string name;
    int age;
};

int main() {
    Student student = {.name = "彭于斌", .age = 23};
    std::string str = serialize(student);
    Student student2 = deserialize(str);
    assert(student.name == student2.name);
    assert(student.age == student2.age);
}
