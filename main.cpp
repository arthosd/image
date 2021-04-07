#define _USE_MATH_DEFINES
#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "src/components/Image.h"
#include <cmath>

using namespace std;
using namespace cv;

int main()
{

    // Exemple de pipeline

    Image image = Image("/home/elie/Documents/Projet/Fac/Image/assets/benjy_jus4.jpeg");
    image.to_gray();
    image.remove_noise(5);
    //image.detect_edge();
    image.apply_gabor(20, 1, M_1_PI / 4, M_1_PI / 1, 0.02, 0); // Fonction de gabor
    image.show("detect");

    return 0;
}
