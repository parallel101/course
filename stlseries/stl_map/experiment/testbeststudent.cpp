#include <algorithm>
#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
using namespace std;

int main() {
    map<int, string> score = {
        {100, "彭于斌"},
        {80, "樱花粉蜜糖"},
        {0, "相依"},
        {60, "Sputnik02"},
    };
    string poorestStudent = score.begin()->second;   // 成绩最差学生的姓名
    string bestStudent = prev(score.end())->second;  // 成绩最好学生的姓名
    print(poorestStudent);
    print(bestStudent);
    return 0;
}
