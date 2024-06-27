void f1(...) {
}

void f2(int i, ...) {
}

void f3(int i...) {
}

void f4(int...) {
}

template <class ...Args>
void f5(Args ...args, ...) {
}

template <class ...Args>
void f6(Args ...args...) {
}

template <class ...Args>
void f7(Args ......) {
}
