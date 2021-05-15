#define _USE_MATH_DEFINES
#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "src/components/Image.h"
#include <cmath>
#include <string.h>
#include <dirent.h>

using namespace std;
using namespace cv;

void pipeline(string PATH)
{
    Image image = Image(PATH);

    image.to_gray();
    //image.show("TO GREY");
    image.remove_noise(5);
    image.detect_edge(20, 120);
    image.show("detect");

    //imshow("Hough", image.hough_transform_prob());
    //waitKey(0);
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

    //apply_pipeline_on_dir("/home/elie/Documents/Projet/Fac/Image/assets/");
    /*Image image = Image("/home/elie/Documents/Projet/Fac/Image/assets/eau3.jpeg");
    image.to_gray();
    image.show("TO GREY");
    image.remove_noise(3);
    image.detect_edge(100, 120);
    image.show("detect");
    imshow("Hough", image.hough_transform_prob());
    waitKey(0);*/

    return 0;
}
