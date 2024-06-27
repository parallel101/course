// https://en.cppreference.com/w/cpp/language/operator_alternative

/*
and    &&
and_eq &=
bitand &
bitor  |
compl  ~
not    !
not_eq !=
or     ||
or_eq  |=
xor    ^
xor_eq ^=
{     <%
}     %>
[     <:
]     :>
#     %:
##    %:%:
*/

%:include <iostream> // #include <iostream>
%:include <string>

void f1(int bitand i) <% // f1(int & i) 左值引用
    i = 42;
%>

void f2(std::string and s) <% // f2(std::string && s) 右值引用
    std::cout << s << '\n';
%>

struct C <%
    compl C() <%  // ~C() { ... }
        std::cout << "C 析构了\n";
    %>
%>;

int main() <%
    C c;
    int num = 10;
    f1(num);
    std::cout << num << '\n';
    f2("hello");
    int arr<:4:> = <%1, 2, 3, 4%>;
    auto lambda = <:bitand:> () -> void <% // [&] -> void { ... arr[i] ... }
        std::cout << arr<:1:> << '\n';
    %>;
    lambda();
    return 0;
%>
