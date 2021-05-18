#include "Image.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
using std::vector;

/*
    Applique sobel
*/
void Image::sobel(int kernel_size, int scale, int delta)
{
    Mat grad_y;

    Sobel(this->image, this->image, CV_8U, 0, 1, kernel_size, scale, delta, BORDER_DEFAULT);
}

Mat Image::hough_transform_prob(int tresh, int rho)
{
    vector<Vec4i> lines; // Contient les lignes qu'on va récupérer

    Mat cdst;

    HoughLinesP(this->image, lines, 1, CV_PI / 2, tresh); // Lance la transformé

    cvtColor(this->image, cdst, COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    return cdst;
}

Mat Image::hough_transform(int tresh)
{
    // Va contenir les traits de l'image
    Mat image = Mat::zeros(this->height, this->width, CV_8UC1);
    vector<Vec2f> lines; //Contient les lignes qu'on va récupérer
    Mat dst;

    HoughLines(this->image, lines, 2, CV_PI / 2, tresh); // Lance la detection des lignes

    // cvtColor(this->image, dst, COLOR_GRAY2BGR);
    cvtColor(image, dst, COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i][1] != 0)
        {
            float rho = lines[i][0], theta = lines[i][1];
            cout << lines[i][1] << endl;
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            cout << pt1 << endl;
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));

            line(dst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
        }
    }

    return dst;
}
/*
    Clusterise l'image en utilisant K-mean
*/
void Image::cluster(int nb_cluster, vector<float> histRows, int max_value)
{

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

    /* // Tablea pour KMEAN
    Mat label, centers;

    // Image en niveau de gris
    if (is_colored == 0)
    {
        // Les données à faire passer au KMEAN
        Mat data(this->height, this->width, CV_32FC(3));

        for (int x = 0; x < this->height; x++)
        {
            for (int y = 0; y < this->width; y++)
            {
                // Intenisté de niveau de gris
                data.at<Vec<float, 3>>(x, y)[0] = this->image.at<float>(x, y) / 255;

                // Coordonnées du pixel
                data.at<Vec<float, 3>>(x, y)[1] = ((float)x) / this->height;
                data.at<Vec<float, 3>>(x, y)[2] = ((float)y) / this->width;

                cout << "c'est pas ciao" << endl;
            }
        }

        cout << "c'est ciao" << endl;

        data = data.reshape(1, data.total());

        // On clusterise
        kmeans(data, nb_cluster, label, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
    }
    // Image en couleur
    else
    {
    }*/
}
/*
    Calcul l'histogramme projeté
*/
Mat Image::calculate_projected_histogram_cropped()
{

    // Vector de données
    std::vector<float> histRows(this->image.rows, 0.0);

    int max = 0;

    for (int i = 0; i < this->image.rows; i++)
    {
        for (int j = 0; j < this->image.cols; j++)
        {
            if (this->image.at<uchar>(i, j) != 0)
            {
                histRows.at(i)++;
            }
        }

        if (histRows.at(i) > max)
        {
            max = histRows.at(i); //Valeur maximale de l'histogramme de projection
        }
    }

    Mat imgHist = Mat::zeros(this->image.rows, this->image.cols, CV_8UC1);

    for (int i = 0; i < this->image.rows; i++)
    {
        if (((float)histRows.at(i) / (float)max) >= 0.4)
        {
            for (int j = 0; j < (int)((histRows.at(i) / max) * imgHist.cols); j++)
            {
                imgHist.at<uchar>(i, j) = 255;
            }
        }
    }

    return imgHist;
}

Mat Image::calculate_projected_histogram()
{
    Mat proj = Mat::zeros(this->height, this->width, CV_8UC1);

    int compteur = 0;
    cout << compteur << endl;

    for (int x = 0; x < this->height; x++)
    {
        compteur = 0;

        for (int y = 0; y < this->width; y++)
        {
            int i = (int)this->image.at<uchar>(x, y);

            if (i > 200)
            {
                cout << (int)this->image.at<uchar>(x, y) << endl;
                proj.at<uchar>(x, compteur) = 255;
                compteur++;
            }
        }
    }

    return proj;
}
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
    Calcul les contours de l'image
*/
void Image::detect_edge(int dt, int ut)
{
    Mat temp_image;
    Canny(this->image, temp_image, dt, ut);
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
    Récupère la valeur du pixel gris à la position X et Y
*/
float Image::get_grey(int x, int y)
{
    Vec3b intensity = this->image.at<Vec3b>(x, y);

    return intensity.val[0];
}
/*
    Retourne l'histogramme projeté de l'image
*/
Mat Image::get_projected_histogram()
{
    return this->projected_histogram;
}
void Image::show_projected_histogram()
{
    imshow("Histograme projeté", this->histogram_projected);
    waitKey(0);
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