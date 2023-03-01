#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
using namespace std;

struct Student {
    int id;             // 学号
    int age;            // 年龄
    string sex;         // 性别
    int money;          // 存款
    set<string> skills; // 技能
};

DEF_PRINT(Student, id, age, sex, money, skills);

map<string, Student> stus = {
    {"彭于斌", {20220301, 22, "自定义", 1000, {"C", "C++"}}},
    {"相依", {20220302, 21, "男", 2000, {"Java", "C"}}},
    {"樱花粉蜜糖", {20220303, 20, "女", 3000, {"Python", "CUDA"}}},
    {"Sputnik02", {20220304, 19, "男", 4000, {"C++"}}},
};

void PeiXunCpp(string stuName) {
    Student stu = stus.at(stuName);
    stu.money -= 2650;
    stu.skills.insert("C++");
}

int main() {
    PeiXunCpp("彭于斌");
    print(stus.at("彭于斌"));
}
