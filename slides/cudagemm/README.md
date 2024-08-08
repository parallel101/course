CMake 报错找不到 CUDA？添加环境变量：

```bash
export PATH="/opt/cuda/bin:$PATH"    # 这里换成你的 cuda 安装位置
export NVCC_CCBIN="/usr/bin/g++-13"  # 只有 Arch Linux 需要这一行
```

Clangd 报错不认识 -forward-unknown-to-host-compiler 选项？

创建文件 `~/.config/clangd/config.yaml`：

```yaml
CompileFlags:
  Add:
    - --cuda-gpu-arch=sm_86
  Remove:
    - -forward-unknown-to-host-compiler
    - --generate-code=*
    - -arch=*
```
