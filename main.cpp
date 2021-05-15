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

    Image image = Image("/home/elie/Documents/Projet/Fac/Image/assets/eau3.jpeg");
    image.to_gray();
    image.show("TO GREY");
    image.remove_noise(3);
    image.detect_edge(100, 120);
    image.show("detect");
    imshow("Hough", image.hough_transform_prob());
    waitKey(0);

    return 0;
}
