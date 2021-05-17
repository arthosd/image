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

Mat Image::hough_transform_prob(int tresh)
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
    vector<Vec2f> lines; //Contient les lignes qu'on va récupérer
    Mat dst;
    HoughLines(this->image, lines, 2, CV_PI / 2, tresh); // Lance la detection des lignes

    cvtColor(this->image, dst, COLOR_GRAY2BGR);

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
void Image::cluster(int nb_cluster, int is_colored)
{
    // Tablea pour KMEAN
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
    }
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
/*

*/
Mat Image::calculate_projected_histogram()
{
    Mat proj(this->height, this->width, CV_8UC1);

    int compteur = 0;

    for (int x = 0; x < this->height; x++)
    {
        compteur = 0;

        for (int y = 0; y < this->width; y++)
        {
            int i = this->image.at<uchar>(x, y);

            if (i > 200)
            {
                cout << (int)this->image.at<uchar>(x, y) << endl;
                proj.at<uchar>(x, compteur) = 255;
                compteur++;
            }
        }
    }

    imshow("SHHHH", proj);
    waitKey(0);

    return proj;
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