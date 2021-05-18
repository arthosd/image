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

void hough_pipeline(String PATH)
{
    Image image = Image(PATH);

    image.to_gray();
    image.remove_noise(5);

    // On detect les contour de l'image
    image.detect_edge(20, 150);
    image.show("Canny");

    // On cacule l'histogramme projeté
    //image.calculate_projected_histogram();

    //imshow("Projeted - Image", proj);
    //waitKey(0);

    // Calculer la transformé de hough pour des valeurs de 90 degrées
    //Mat hough = image.hough_transform(100);

    //imshow("Hough - Transform", hough);
    //waitKey(0);
}

void pipeline(string PATH)
{
    Image image = Image(PATH);

    image.to_gray();
    image.remove_noise(5);

    // On detect les contour de l'image
    image.detect_edge(20, 150);
    image.show("Canny");

    // On cacule l'histogramme projeté
    Mat proj = image.calculate_projected_histogram_cropped();

    imshow("Projecterd - Image", proj);
    waitKey(0);

    // Calculer la transformé de hough pour des valeur de 90 degrées
    Mat hough = image.hough_transform(100);

    imshow("Hough - Transform", hough);
    waitKey(0);
}

int apply_pipeline_on_dir(char *PATH)
{
    DIR *source;

    struct dirent *entry;

    source = opendir(PATH);

    if (source == NULL) // S'il y a une erreur dans l'ouverture du dossier
    {
        cout << "ERREUR DURANT L'OUVERTURE DU DOSSIER SOURCE !" << endl;
        return -1;
    }

    entry = readdir(source);

    while (entry != NULL)
    {

        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
        {
            string file_path = string(PATH) + "/" + string(entry->d_name);
            pipeline(file_path);
        }
        entry = readdir(source);
    }

    return 0;
}

int main()
{

    // Exemple de pipeline

    // apply_pipeline_on_dir("/home/elie/Documents/Projet/Fac/Image/assets/");
    Image image = Image("/home/elie/Documents/Projet/Fac/Image/assets/eau3.jpeg");

    image.to_gray();
    image.remove_noise(5);
    image.detect_edge(20, 150);
    image.show("Canny - Image");

    // Image des lignes de niveaux
    Mat proj = image.calculate_projected_histogram_cropped();
    vector<int> lignes = image.treat_histogram(proj);

    for (int i = 0; i < lignes.size(); i++)
    {
        cout << lignes[i] << endl;
    }

    return 0;
}