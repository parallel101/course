# 现代 C++ 的错误处理

```cpp
try {
    // 一些可能抛出异常的操作
} catch (exception const &e) {
    cout << "发现异常: " << e.what() << '\n';
} catch (...) {
    cout << "未知的异常类型\n";
}
```
