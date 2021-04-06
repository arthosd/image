#include "Image.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
using std::vector;

/*
    Applique une transformée de gabor avec kernel_size comme taille de filtre
*/
void Image::apply_gabor(int kernel_size, double sigma, double theta, double lamda, double gamma, double phi)
{
    Mat temps_image;
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size),
                                        sigma,
                                        theta,
                                        lamda,
                                        gamma,
                                        phi);

    cv::filter2D(this->image, temps_image, CV_32F, kernel);
    imshow("Kernel Gabor", kernel);
    this->image = temps_image;
}
/*
    Calcul les contours de l'image
*/
void Image::detect_edge()
{
    Mat temp_image;
    Canny(this->image, temp_image, 0, 210);
    this->image = temp_image;
}

/*
    Egalise l'histogramme de l'image
*/
void Image::equalize()
{
    Mat temp_image;
    equalizeHist(this->image, temp_image);
    this->image = temp_image;
}

/*
    Effectue un filtre median en utilisant une matrice de taille ksize
*/
void Image::remove_noise(int ksize)
{
    Mat temp_image;
    medianBlur(this->image, temp_image, ksize);
    this->image = temp_image;
}

/*
    Applique le seuillage d'otsu sur l'image
*/
void Image::otsu()
{
    Mat temp_image;
    threshold(this->image, temp_image, 0, 255, THRESH_OTSU);
    this->image = temp_image;
}

/*
    Seuille l'image en fonction du min données en paramètre
*/
void Image::binarize(int min)
{
    Mat temp_image;                                              // Image temporaire
    threshold(this->image, temp_image, min, 255, THRESH_BINARY); // On seuille l'image
    this->image = temp_image;
}

/*
    Convertit l'image en niveau de gris
*/
void Image::to_gray()
{
    Mat temp_image;
    cvtColor(this->image, temp_image, COLOR_BGR2GRAY);
    this->image = temp_image;
}

/*
    Affiche l'image dans une fenetre
*/
void Image::show(string windows_name)
{
    imshow(windows_name, this->image);
    waitKey(0);
}

/*
    Récupère la valeur du pixel gris à la position X et Y
*/
float Image::get_grey(int x, int y)
{
    Vec3b intensity = this->image.at<Vec3b>(x, y);

    return intensity.val[0];
}

/*
    Constructeur de l'image
*/
Image::Image(string image_path)
{
    this->image = imread(image_path);         // On lis l'image
    this->height = this->image.size().height; // La hauteur de l'image
    this->width = this->image.size().width;   // La largeur de l'image
    this->image_path = image_path;            // Le chemin vers l'image
    this->histogramme = new int[255];         // l'Histogramme de l'image
}