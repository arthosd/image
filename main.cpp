#define _USE_MATH_DEFINES

#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "src/components/Image.h"
#include <cmath>
#include <string.h>
#include <dirent.h>
#include <vector>

using namespace std;
using namespace cv;

int main()
{

    Image image = Image("/home/elie/Documents/Projet/Fac/Image/assets/coca3.jpeg");

    image.to_gray(); // Transformation en niveua de gris
    image.show("Image - gris");
    image.remove_noise(5); // On attenue l'image avec un filtre gaussien

    image.detect_edge(20, 120); // On detecte les contours du verre
    image.show("Canny - filter");

    Mat proj = image.line_level(); // On les lignes de niveaux

    imshow("Lignes de niveaux", proj);

    int verify = image.treat_histogram(proj);           // Récupération de la ligne de niveau qui nous interesse
    int houghline = image.hough_transform(100, verify); // Vérification de cette ligne de niveau avec les lignes de la transformée de hough

    image.show("Level - Water");
}
