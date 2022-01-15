#include "Course.h"

void func(Course course) {
    course.func();
}

int main() {
    Course co;
    func(co);
    return 0;
}
