struct A {
    static inline int other(int a) {
        return a;
    }
};

int func() {
    return A().other(233);
}
