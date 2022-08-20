int slow(int x) {
    if (x > 0) {
        return 42;
    } else {
        return 32;
    }
}

int fast(int x) {
    return 32 + (x > 0) * 10;
}
