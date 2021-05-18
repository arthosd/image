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
    image.detect_edge(20, 150);
    image.show("Canny - Image");

    // Image des lignes de niveaux
    Mat proj = image.calculate_projected_histogram_cropped();

    // On rècuperes les lignes représentatives
    int ligne = image.treat_histogram(proj);

    int houghLine = image.hough_transform(120, ligne);

    cout << "------------------------" << endl;
    cout << houghLine << endl;
    cout << ligne << endl;
}

void pipeline(string PATH)
{
    Image image = Image(PATH);

    image.to_gray();
    image.remove_noise(5);

    // On detecte les contours
    image.detect_edge(20, 120);

    // On calcul l'histogramme des lignes de niveaux
    Mat proj = image.calculate_projected_histogram_cropped();

    // on récupère la ligne de niveau qui potentiellement représente le niveua de l'eau
    int verify = image.treat_histogram(proj);

    // On vérifie si la lgine est bonne et si oui elle correspond à quoi
    //int houghline = image.hough_transform(120, verify);

    //cout << verify << endl;
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

    /*Mat image = imread("/home/elie/Documents/Projet/Fac/Image/assets/benjy_eau2.jpeg");
    cvtColor(image, image, COLOR_BGR2GRAY);
    imshow("Image", image);

    for (int y = 0; y < image.rows; y++)
    {
        cout << "y =" << y << endl;
        for (int x = 0; x < image.cols; x++)
        {
            int i = (int)image.at<uchar>(x, y);
            cout << "\t"
                 << "x =" << x << endl;
        }
    }

    cout << "rows = " << image.rows << endl;
    cout << "cols = " << image.cols << endl;

    // Théoriquement : image  */

    //apply_pipeline_on_dir("/home/elie/Documents/Projet/Fac/Image/assets/");
    Image image = Image("/home/elie/Documents/Projet/Fac/Image/assets/jus1.jpeg");

    // On prépare l'image
    image.to_gray();
    image.remove_noise(5);

    // On detecte les contours
    image.detect_edge(20, 120);

    // On calcul l'histogramme des lignes de niveaux
    Mat proj = image.calculate_projected_histogram_cropped();

    // on récupère la ligne de niveau qui potentiellement représente le niveua de l'eau
    int verify = image.treat_histogram(proj);

    // On vérifie si la lgine est bonne et si oui elle correspond à quoi
    int houghline = image.hough_transform(120, verify);

    cout << verify << endl;
    cout << houghline << endl;

    image.show("string");
}