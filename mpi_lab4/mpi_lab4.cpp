#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;

const int n = 7;

tuple<int, int, int> Mean(vector<int> r, vector<int> g, vector<int> b)
{
    sort(r.begin(), r.end());
    sort(g.begin(), g.end());
    sort(b.begin(), b.end());

    int mean = r.size() / 2 + 1;

    //cout << r.size() << ' ' << mean << endl;

    return make_tuple(r[mean], g[mean], b[mean]);
}

Mat Filter(Mat image)
{
    Mat temp = image.clone();

    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
        {
            vector<int> r;
            vector<int> g;
            vector<int> b;

            for (int k = i - 3; k < i + 4; k++)
                for (int l = j - 3; l < j + 4; l++)
                {
                    int x = k, y = l;

                    if (k < 0)
                        x = 0;
                    else
                        if (k >= image.rows)
                            x = image.rows - 1;

                    if (l < 0)
                        y = 0;
                    else
                        if (l >= image.cols)
                            y = image.cols - 1;

                    r.push_back(image.at<Vec3b>(x, y)[0]);
                    g.push_back(image.at<Vec3b>(x, y)[1]);
                    b.push_back(image.at<Vec3b>(x, y)[2]);
                }
            int mean_r, mean_g, mean_b;
            tie(mean_r, mean_g, mean_b) = Mean(r, g, b);

            temp.at<Vec3b>(i, j)[0] = mean_r;
            temp.at<Vec3b>(i, j)[1] = mean_g;
            temp.at<Vec3b>(i, j)[2] = mean_b;
        }

    return temp;
}

int main()
{
    string file_name = "1.jpg";
    Mat image = imread(file_name, IMREAD_COLOR);
    imshow("original", image);
    waitKey(0);


    /*uchar* temp1 = new uchar[image.rows*image.cols*image.channels()];
    int* temp = new int[image.rows * image.cols * image.channels()];
    temp1 = image.;

    temp = */

    vector<uchar> temp(image.rows * image.cols * image.channels());
    temp.assign(image.datastart, image.dataend);

    //int* temp = a.data();

    /*cout << endl << endl;

    int n = image.rows * image.cols * image.channels();
    for (int i = 0; i < n; i += 3)
    {
        int r = temp[i];
        int g = temp[i + 1];
        int b = temp[i + 2];
        cout << "(" << r << "," << g << "," << b << ") ";

        if ((i + 3) % (image.cols * 3) == 0)
            cout << endl << endl << endl;
    }*/

    /*Mat oof(image.rows, image.cols, CV_8UC3);
    oof.data = temp.data();

    imshow("oof", oof);
    waitKey(0);*/

    return 0;
}