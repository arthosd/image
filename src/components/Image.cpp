#include "Image.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
using std::vector;

/*
    Applique la transformé dee hough afin de detecté les lignes horizontales.
    On vérifie si la ligne de niveaux correspond à une des lignes données par Hough.

    int tresh -> Le seuil minimale de l'accumulateur d'une droite
    int number_to_check -> la coordonées Y de la ligne de niveau à vérifier

    Retourne l'image avec la ligne de niveau.
*/
int Image::hough_transform(int tresh, int number_to_check)
{
    int ligne = number_to_check;
    int pas = (10 * number_to_check) / 100;

    vector<Vec2f> lines; //Contient les lignes qu'on va récupérer

    HoughLines(this->image, lines, 2, CV_PI / 2, tresh); // Lance la detection des lignes

    for (size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i][1] != 0)
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));

            for (int i = pt1.y - pas; i < pt1.y + pas; i++)
            {
                if (i == ligne)
                {
                    cvtColor(this->image, this->image, CV_8UC1);
                    line(this->image, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
                    return pt1.y;
                }
            }
        }
    }

    return -1;
}
/*
    Clusterise l'image en utilisant K-mean
*/
void Image::cluster(int nb_cluster, vector<float> histRows, int max_value)
{

    /////////// CETTE FONCTION N'EST PAS FONCTIONNELLE.

    // Labels et centroids
    Mat labels, centers;

    // Data à clusteriser
    Mat data = Mat::zeros(this->height, 1, CV_32FC2);

    for (int i = 0; i < histRows.size(); i++)
    {
        if (((float)histRows.at(i) / (float)max_value) >= 0.38)
        {
            data.at<float>(i, 0) = histRows.at(i) / (float)max_value;
        }
        // A retirer psk c'est déjà une matrice rempli de Zéros ----------------------------
        else
        {
            data.at<float>(i, 0) = 0;
        }

        data.at<float>(i, 1) = (float)i / (float)(histRows.size() - 1);
    }

    int firstRow = 0;
    int lastRow = (int)histRows.size() - 1;

    for (int i = 0; data.at<float>(i, 0) == 0; i++)
    {
        firstRow++;
    }
    for (int i = histRows.size() - 1; data.at<float>(i, 0) == 0; i--)
    {
        lastRow--;
    }

    std::vector<float> histRowsCropped((lastRow - firstRow) + 1, 0.0);

    for (int i = firstRow; i <= lastRow; i++)
    {
        histRowsCropped.at(i - firstRow) = histRows.at(i);
    }

    Mat data2 = Mat::zeros(histRowsCropped.size(), 1, CV_32FC2);

    for (int i = 0; i < histRowsCropped.size(); i++)
    {
        if (((float)histRowsCropped.at(i) / (float)max_value) >= 0.4)
        {
            data2.at<float>(i, 0) = histRowsCropped.at(i) / (float)max_value;
        }
        else
        {
            data2.at<float>(i, 0) = 0;
        }
        data2.at<float>(i, 1) = (float)i / (float)(histRowsCropped.size() - 1);
    }

    kmeans(data2, 4, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 3, 1.0), 10, KMEANS_PP_CENTERS, centers);

    std::vector<int> sortedCenterRows(4, 0);

    for (int i = 0; i < 4; i++)
    {
        sortedCenterRows[i] = ((int)(centers.at<float>(i, 1) * histRowsCropped.size())) + firstRow;
    }

    std::sort(sortedCenterRows.begin(), sortedCenterRows.begin() + 4);

    // Va contenir le résultat
    Mat Result = Mat::zeros(this->height, this->width, CV_8UC1);

    for (int j = 0; j < Result.cols; j++)
    {
        Result.at<Vec3b>(sortedCenterRows.at(1), j) = Vec3b(255, 0, 0);
    }
    for (int j = 0; j < Result.cols; j++)
    {
        Result.at<Vec3b>(sortedCenterRows.at(2), j) = Vec3b(0, 0, 255);
    }

    imshow("Image", Result);
    waitKey(0);
}
/*
    Calcul les lignes de niveaux d'une image.

    Retourne l'histogramme représantant les lignes de niveaux.
*/
Mat Image::line_level()
{

    // Vector de données
    std::vector<float> rows(this->image.rows, 0.0);

    int max = 0;

    for (int i = 0; i < this->image.rows; i++)
    {
        for (int j = 0; j < this->image.cols; j++)
        {
            if (this->image.at<uchar>(i, j) != 0)
            {
                rows.at(i)++;
            }
        }

        if (rows.at(i) > max)
        {
            max = rows.at(i); //Valeur maximale de l'histogramme de projection
        }
    }

    Mat hist = Mat::zeros(this->image.rows, this->image.cols, CV_8UC1);

    for (int i = 0; i < this->image.rows; i++)
    {
        if (((float)rows.at(i) / (float)max) >= 0.4)
        {
            for (int j = 0; j < (int)((rows.at(i) / max) * hist.cols); j++)
            {
                hist.at<uchar>(i, j) = 255;
            }
        }
    }

    return hist;
}
/*
    Calcul l'histogramme projeté de l'image
*/
Mat Image::calculate_projected_histogram()
{
    Mat proj = Mat::zeros(this->height, this->width, CV_8UC1);

    int compteur = 0;

    for (int x = 0; x < this->height; x++)
    {
        compteur = 0;

        for (int y = 0; y < this->width; y++)
        {
            int i = (int)this->image.at<uchar>(x, y);

            if (i > 200)
            {
                proj.at<uchar>(x, compteur) = 255;
                compteur++;
            }
        }
    }

    return proj;
}
/*
    Découpe les lignes de niveaux en trois zones afin d'en extraire trois lignes de niveaux significatifs.

    Mat proj -> Prend l'hitogramme représantant les lignes de niveaux

    Renvoie la ligne représentant le candidat parmit toutes les lignes de niveaux.
*/
int Image::treat_histogram(Mat proj)
{

    std::vector<int> total(proj.rows, 0);
    std::vector<int> lignes(3, 0);

    int choix_ligne = -1;

    for (int x = 0; x < proj.rows; x++)
    {
        for (int y = 0; y < proj.cols; y++)
        {
            if (proj.at<uchar>(x, y) > 200)
                total[x]++;
        }
    }

    int pas = total.size() / 3; // On divise par trois la taille de l'image projeté

    int debut = 0;         // Le début de la zone
    int fin = debut + pas; // La fin de la zone

    for (int compteur = 0; compteur < pas; compteur++)
    {
        for (int i = debut; i < fin; i++)
        {
            if (total[i] != 0)
            {
                int temp = total[i];

                if (compteur == 0)
                {
                    if (temp > lignes[compteur])
                        lignes[compteur] = i;
                }
                else
                {
                    // On cherche le plus petit
                    if (lignes[compteur] == 0)
                    {
                        lignes[compteur] = i;
                    }
                    else
                    {
                        if (temp < lignes[compteur])
                            lignes[compteur] = i;
                    }
                }
            }
        }

        debut = fin;
        fin = debut + pas;
    }

    // On choisi la ligne de niveau qui représente l'eau

    if (lignes[1] != 0)
        choix_ligne = lignes[1];

    else
        choix_ligne = lignes[0];

    return choix_ligne;
}
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
    cv::normalize(kernel, kernel, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::resize(kernel, kernel, cv::Size(), 6, 6);
    imshow("Kernel Gabor", kernel);
    cv::normalize(temps_image, this->image, 0, 255, NORM_MINMAX, CV_8UC1);
}
/*
    Calcul les contours de l'image en utilisants Canny
*/
void Image::detect_edge(int dt, int ut)
{
    Mat temp_image;
    Canny(this->image, temp_image, dt, ut);
    this->image = temp_image;
}
/*
    Effectue un filtre Gaussen en utilisant une matrice de taille ksize
*/
void Image::remove_noise(int ksize)
{
    Mat temp_image;
    GaussianBlur(this->image, temp_image, Size(ksize, ksize), 0);
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
    Set la valeur d'un pixel
*/
void Image::set_grey(cv::Mat image, int x, int y, int value)
{
    image.at<Vec3b>(x, y) = value; // Set la valeur de l'image
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
}
Image::Image(Mat image)
{
    this->image = image;                      // On lis l'image
    this->height = this->image.size().height; // La hauteur de l'image
    this->width = this->image.size().width;   // La largeur de l'image
    this->image_path = image_path;            // Le chemin vers l'image
}