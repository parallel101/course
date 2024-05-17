在 co_async 的 15000 行源码中，`new` 只出现了一次：

```cpp
new (std::addressof(mValue)) T(std::forward<Ts>(args)...);
```

在 2100 行的 `<GL/gl.h>` 头文件中，`struct` 一次都没出现。
