void other();

float func(float *a) {
    float ret = 0;
    for (int i = 0; i < 1024; i++) {
        ret += a[i];
        other();
    }
    return ret;
}
