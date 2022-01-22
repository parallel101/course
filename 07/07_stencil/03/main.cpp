#include <iostream>
#include "Image.h"
#include "rwimage.h"
#include "boxblur.h"
#include "gaussblur.h"
#include "ticktock.h"

int main() {
    Image a;
    TICK(read);
    read_image(a, "original.jpg");
    TOCK(read);

    //TICK(boxblur);
    //boxblur(a, 32, 32);
    //TOCK(boxblur);

    TICK(gaussblur);
    gaussblur(a, 32, 12);
    TOCK(gaussblur);

    TICK(write);
    write_image(a, "result.png");
    TOCK(write);
    return 0;
}
