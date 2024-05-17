#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
using namespace std;

struct Student {
    string name;
    int id;
    string sex;
};
DEF_PRINT(Student,, name, id, sex);

struct LessStudent {
    bool operator()(Student const &x, Student const &y) const {
        return x.name < y.name; // || x.id < y.id || x.sex < y.sex;
    }
};

int main() {
    map<Student, int, LessStudent> stutab = {
        {Student{"小彭老师", 233, "自定义"}, 100},
        {Student{"相依", 985, "男"}, 99},
        {Student{"樱花粉蜜糖", 211, "女"}, 42},
    };
    auto it = stutab.find(Student{"小彭老师", 0, ""});
    if (it != stutab.end()) {
        print("找到", it->first, it->second);
    } else {
        print("没找到");
    }
    return 0;
}
