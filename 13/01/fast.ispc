export void simple(uniform float dst[], uniform float src[]) {
    foreach (i = 0 ... 4096) {
        float r = src[i * 3];
        float g = src[i * 3 + 1];
        float b = src[i * 3 + 2];
        dst[i] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}

