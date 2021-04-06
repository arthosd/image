#ifndef DEF_IMAGE
#define DEF_IMAGE

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class Image
{

public:
    Image(std::string image_path);       // Constructeur
    void to_gray();                      // Convertit l'image en niveau de gris
    void show(std::string windows_name); // Affiche l'image
    void binarize(int min);              // Effectue une seuillage sur l'image
    void otsu();                         // Effectue un seuillage d'otsu
    void remove_noise(int ksize);        // Effectue un filtre médian pour supprimer le bruit
    void equalize();                     // Egalise l'histogramme de l'image
    void detect_edge();                  // Trouve les contours de l'image
    void apply_gabor(int kernel_size,    // On applique gabor sur l'image
                     double sigma,
                     double theta,
                     double lambda,
                     double gamma,
                     double phi);

private:
    // Attibuts
    cv::Mat image;          // L'image
    std::string image_path; // Le chemin vers l'image
    int width, height;      // Dimensions de l'image
    int *histogramme;       // Histogramme de l'image

    // Fonctions
    float get_grey(int x, int y); //Rècupère le niveau de gris d'un pixel à la position X et Y
};
#endif