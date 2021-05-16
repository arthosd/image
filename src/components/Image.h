#ifndef DEF_IMAGE
#define DEF_IMAGE

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class Image
{

public:
    Image(std::string image_path);                         // Constructeur - 1
    Image(cv::Mat image);                                  // Constructeur - 2
    void set_grey(cv::Mat image, int x, int y, int value); // Set la valeur du pixel  à la valeur données
    cv::Mat calculate_projected_histogram();               // calcul et affiche l'histogramme projeté de l'image
    void to_gray();                                        // Convertit l'image en niveau de gris
    void show(std::string windows_name);                   // Affiche l'image
    void binarize(int min);                                // Effectue un seuillage sur l'image
    void otsu();                                           // Effectue un seuillage d'otsu
    void remove_noise(int ksize);                          // Effectue un filtre médian pour supprimer le bruit
    void equalize();                                       // Egalise l'histogramme de l'image
    void detect_edge(int dt, int ut);                      // Trouve les contours de l'image
    void cluster(int nb_cluster);                          // Cluster l'image en utilisant K-Mean
    void sobel(int kernel_size, int scale, int delta);     // Filter de sobel
    cv::Mat hough_transform(int tresh);                    // Calcul la transformé de hough
    cv::Mat hough_transform_prob(int tresh);               // Calcul la transformé de hough probabiliste
    cv::Mat get_projected_histogram();                     // Renvoie l'histogramme projeté de l'image
    void show_projected_histogram();                       // Affiche l'histogramme projeté de l'image
    void apply_gabor(int kernel_size,                      // On applique gabor sur l'image
                     double sigma,
                     double theta,
                     double lambda,
                     double gamma,
                     double phi);

private:
    // Attibuts
    cv::Mat image;               // L'image
    cv::Mat projected_histogram; // Histogramme projeté de l'image
    std::string image_path;      // Le chemin vers l'image
    int width, height;           // Dimensions de l'image
    cv::Mat histogram_projected; // L'histogramme projeté de l'image

    // Fonctions
    float get_grey(int x, int y); //Rècupère le niveau de gris d'un pixel à la position X et Y
};
#endif