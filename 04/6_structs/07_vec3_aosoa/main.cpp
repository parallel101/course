struct MyVec {
    float x[4];
    float y[4];
    float z[4];
};

MyVec a[1024 / 4];

void func() {
    for (int i = 0; i < 1024 / 4; i++) {
        for (int j = 0; j < 4; j++) {
            a[i].x[j] *= a[i].y[j];
        }
    }
}
