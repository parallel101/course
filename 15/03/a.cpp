#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    char s1[] = "hello";
    char s2[] = "world";
    char *s3 = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(s3, s1);
    strcat(s3, s2);
    printf("%s\n", s3);
    free(s3);
}
