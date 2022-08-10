#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <mpi.h>
using namespace cv;
using namespace std;

const int n = 7;

tuple<int, int, int> Mean(vector<int> r, vector<int> g, vector<int> b)
{
    sort(r.begin(), r.end());
    sort(g.begin(), g.end());
    sort(b.begin(), b.end());

    int mean = r.size() / 2 + 1;

    return make_tuple(r[mean], g[mean], b[mean]);
}

Mat Filter(Mat image)
{
    Mat temp = image.clone();

    for (int i = 0; i < image.rows; i++)
    {
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
    }

    return temp;
}

int main(int argc, char** argv)
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time = MPI_Wtime();

    int rows, cols;
    int* part;

    if (rank == 0)
    {
        string file_name = "1.jpg";
        Mat image = imread(file_name, IMREAD_COLOR);
        imshow("ыв", image);
        waitKey(0);

        int i_rows = image.rows;
        int i_cols = image.cols;

        vector<int> temp((long)i_rows * i_cols * 3);
        temp.assign(image.datastart, image.dataend);

        MPI_Datatype MPI_IMAGE_PART;
        MPI_Type_vector(temp.size() / size, temp.size() / size, 0, MPI_INT, &MPI_IMAGE_PART);
        MPI_Type_commit(&MPI_IMAGE_PART);

        /*MPI_Scatter(temp.data(), temp.size() / size, MPI_IMAGE_PART,
            part, temp.size() / size, MPI_IMAGE_PART, 0, MPI_COMM_WORLD);*/

        MPI_Type_free(&MPI_IMAGE_PART);

        MPI_Bcast(&i_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&i_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    /*MPI_Recv(&rows, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
    MPI_Recv(&cols, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);

    vector<int> temp(part, part + sizeof part / sizeof part[0]);
    vector<uchar> temp1;
    temp1.assign(temp.begin(), temp.end());

    Mat oof(rows / 4, cols, CV_8UC3);
    oof.data = temp1.data();

    imshow(to_string(rank), oof);
    waitKey(0);*/

    //MPI_Type_free(&MPI_IMAGE_PART);
	MPI_Finalize();
	return 0;
}