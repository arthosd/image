#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "src/components/Image.h"

using namespace std;
using namespace cv;

int main()
{

    // Exemple de pipeline

    Image image = Image("/home/elie/Documents/Projet/Fac/Image/assets/eau1.jpeg");
    image.to_gray();
    image.remove_noise(5);
    image.detect_edge();
    image.show("detect");

    return 0;
}
