#include "Image.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
using std::vector;

Mat Image::hough_transform_prob()
{
    vector<Vec4i> lines; // Contient les lignes qu'on va récupérer

    Mat cdst;

    HoughLinesP(this->image, lines, 1, CV_PI / 180, 50, 50, 10); // Lance la transformé

    cvtColor(this->image, cdst, COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    return cdst;
}

Mat Image::hough_transform()
{
    vector<Vec2f> lines; //Contient les lignes qu'on va récupérer
    Mat dst;
    HoughLines(this->image, lines, 1, CV_PI / 180, 150, 0, 0); // Lance la detection des lignes

    cvtColor(this->image, dst, COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(dst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    return dst;
}
/*
    Clusterise l'image en utilisant K-mean
*/
void Image::cluster(int nb_cluster)
{
    Mat data;
    Mat pixel(this->height, this->width, CV_32FC3);
    this->image.convertTo(data, CV_32F); // Convertit l'image en représentation flottante

    /*
        Data -> contient l'image initial en format float
        pixel -> Image vide avec les meme dimension que l'image initial
    */

    // On itère dans l'image flottante initial
    for (int x = 0; x < this->height; x++)
    {
        for (int y = 0; y < this->width; y++)
        {
            pixel.at<Vec<float, 5>>(x, y)[0] = data.at<Vec3f>(x, y)[0] / 255;
            pixel.at<Vec<float, 5>>(x, y)[1] = data.at<Vec3f>(x, y)[1] / 255;
            pixel.at<Vec<float, 5>>(x, y)[2] = data.at<Vec3f>(x, y)[2] / 255;

            pixel.at<Vec3f>(x, y)[3] = ((float)x) / this->height;
            pixel.at<Vec3f>(x, y)[4] = ((float)y) / this->width;
        }
    }

    pixel = pixel.reshape(1, pixel.total());

    Mat labels, centers;

    kmeans(pixel, nb_cluster, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    for (int i = 0; i < data.total(); ++i)
    {
        //cout << labels.at<int>(i) << "  " << centers.at<float>(labels.at<int>(i)) << endl;
        data.at<Vec3f>(i)[0] = centers.at<float>(labels.at<int>(i), 0) * 255;
        data.at<Vec3f>(i)[1] = centers.at<float>(labels.at<int>(i), 0) * 255;
        data.at<Vec3f>(i)[2] = centers.at<float>(labels.at<int>(i), 0) * 255;
    }

    data = data.reshape(this->image.channels(), this->image.rows);
    data.convertTo(data, CV_8U);

    imshow("Cluster", data);
    waitKey(0);
}
/*
    Calcul l'histogramme projeté
*/
void Image::projected_histogram()
{
    // QUi va contenir l'histogramme projeté
    cv::Mat temp(this->width, this->height, CV_8UC3, cv::Scalar(0, 0, 0));
    int compteur = 0; // Compyeur des blancs

    for (int y = 0; y < this->height; y++)
    {
        compteur = 0;

        for (int x = 0; x < this->width; x++)
        {
            Vec3b pixel = this->image.at<Vec3b>(x, y); // On récupère le niveau de gris

            if (pixel.val[0] != 0) // Si le pixel est blanc
            {
                temp.at<Vec3b>(x, y).val[0] = 255;
                compteur++;
            }
        }
    }

    imshow("Projected", temp);
    waitKey(0);
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
    Constructeur de l'image
*/
Image::Image(string image_path)
{
    this->image = imread(image_path);         // On lis l'image
    this->height = this->image.size().height; // La hauteur de l'image
    this->width = this->image.size().width;   // La largeur de l'image
    this->image_path = image_path;            // Le chemin vers l'image
    this->can_project_histogram = false;      //Vérifie s'il y peux projter l'image
}