## 真正的内存模型！

```cpp
void modify(int *pa) {
    int *pb = pa + 1;
    *pb = 9;
}

int func() {
    int a = 4;
    int b = 5;
    modify(&a);
    return b; // 5 还是 9？
}
```
