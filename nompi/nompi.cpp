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

Mat Filter(Mat image, int sc)
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

Mat delete_extra_rows_top(Mat a, int extra_rows_count)
{
    Mat b;

    a(Range(extra_rows_count, a.rows), Range(0, a.cols)).copyTo(b);

    return b;
}

Mat delete_extra_rows_bottom(Mat a, int extra_rows_count)
{
    Mat b;

    a(Range(0, a.rows - extra_rows_count), Range(0, a.cols)).copyTo(b);

    return b;
}

int main(int argc, char** argv)
{
    string num = "1";
    int extra_row_count = n / 2;

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time = MPI_Wtime();

    int rows, cols;
    Mat r, g, b;

    if (rank == 0)
    {
        cout << "image " << num << endl;

        //читаем изображение
        string file_name = num + ".jpg";
        Mat image = imread(file_name, IMREAD_COLOR);
        imshow(file_name, image);
        waitKey(0);

        rows = image.rows;
        cols = image.cols;

        //разделяем изображение на каналы для отправки
        Mat* channels = new Mat[3];
        split(image, channels);

        r = channels[2];
        g = channels[1];
        b = channels[0];

    }

    //рассылаем количества строк и столбцов
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //cout << rank << " " << rows << " " << cols;

    //тип для одной строки
    MPI_Datatype IMAGE_ROW;
    MPI_Type_vector(1, cols, cols, MPI_UNSIGNED_CHAR, &IMAGE_ROW);
    MPI_Type_commit(&IMAGE_ROW);

    int* sendcounts = nullptr, * displs = nullptr;

    sendcounts = new int[size];
    displs = new int[size];

    int part_size = rows / size;

    //задаем размеры отправляемых частей и сдвиги с учетом дополнительных строк
    for (int i = 0; i < size - 1; i++)
    {
        sendcounts[i] = part_size + extra_row_count;
        displs[i] = part_size * i;

        if (i > 0)
        {
            sendcounts[i] += extra_row_count;
            displs[i] -= extra_row_count;
        }
    }

    sendcounts[size - 1] = rows - part_size * (size - 1) + extra_row_count;
    displs[size - 1] = part_size * (size - 1) - extra_row_count;

    Mat r_part(sendcounts[rank], cols, CV_8UC1);
    Mat g_part(sendcounts[rank], cols, CV_8UC1);
    Mat b_part(sendcounts[rank], cols, CV_8UC1);

    //MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(r.data, sendcounts, displs, IMAGE_ROW, r_part.data, sendcounts[rank], IMAGE_ROW, 0, MPI_COMM_WORLD);
    MPI_Scatterv(g.data, sendcounts, displs, IMAGE_ROW, g_part.data, sendcounts[rank], IMAGE_ROW, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b.data, sendcounts, displs, IMAGE_ROW, b_part.data, sendcounts[rank], IMAGE_ROW, 0, MPI_COMM_WORLD);

    //собираем ргб каналы в одно изображение
    Mat image_part;
    vector<Mat> chs{ b_part, g_part, r_part };
    merge(chs, image_part);
    
    image_part = Filter(image_part, sendcounts[rank]);

    //удаляем лишние строки
    if (rank > 0)
        image_part = delete_extra_rows_top(image_part, extra_row_count);
    if (rank < size - 1)
        image_part = delete_extra_rows_bottom(image_part, extra_row_count);

    //разделяем на каналы для отправки
    Mat* channels = new Mat[3];
    split(image_part, channels);

    Mat new_r = channels[2];
    Mat new_g = channels[1];
    Mat new_b = channels[0];

    //перерасчет размеров частей и сдвигов, с учетом удаленных лишних строк
    for (int i = 0; i < size; i++)
    {
        if (i > 0)
        {
            sendcounts[i] -= extra_row_count;
            displs[i] += extra_row_count;
        }

        if (i < size - 1)
            sendcounts[i] -= extra_row_count;
    }

    MPI_Gatherv(new_r.data, new_r.rows, IMAGE_ROW, r.data, sendcounts, displs, IMAGE_ROW, 0, MPI_COMM_WORLD);
    MPI_Gatherv(new_g.data, new_g.rows, IMAGE_ROW, g.data, sendcounts, displs, IMAGE_ROW, 0, MPI_COMM_WORLD);
    MPI_Gatherv(new_b.data, new_b.rows, IMAGE_ROW, b.data, sendcounts, displs, IMAGE_ROW, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        Mat new_image;
        vector<Mat> chs{ b, g, r };
        merge(chs, new_image);

        string file_name = num + "_new_par.jpg";

        imshow(file_name, new_image);
        waitKey(0);
        //imwrite(file_name, new_image);
    }

    time = MPI_Wtime() - time;

    cout << endl << "rank: " << rank << " time = " << time << endl;

    double max_time;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "\ntotal time = " << max_time << endl;

    delete[]sendcounts;
    delete[]displs;

    MPI_Type_free(&IMAGE_ROW);
    MPI_Finalize();
    return 0;
}