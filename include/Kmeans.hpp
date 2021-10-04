#pragma once

#include <vector>

class Kmeans
{
private:
    /* data */
    int number_clusters;
    int dimension_data;

    

public:

    std::vector<std::vector<double> > centroid;

    Kmeans(int num_features, int k);
    ~Kmeans();

    void Initialize(int k_cluster, std::vector< std::vector<double> > data);
    int Classify(std::vector<double> &data);
    double Cluster(int number_data, std::vector< std::vector<double> > data);
};

/**
 * @brief Construct a new Kmeans:: Kmeans object
 * 
 * @param dimension_data    M dimension of the data matrix
 * @param number_clusters   k how many cluster should be calculated
 */
Kmeans::Kmeans(int dimension_data, int number_clusters)
{
    this->dimension_data = dimension_data;
    this->number_clusters = number_clusters;

    centroid.resize(number_clusters);

    for(int i = 0; i < number_clusters; i++)
    {
        centroid[i].resize(dimension_data);
    }
}

Kmeans::~Kmeans()
{
}

/**
 * @brief Initialize Kmeans Cluster with randowm poicked centroids
 * 
 * @param number_data   N number of data
 * @param data          N x M shaped Matrix of data
 */
void Kmeans::Initialize(int number_data, std::vector<std::vector<double> > data)
{
    // Divide data into cluster bins
    int number_sample = number_data / number_clusters;
	
	for(int i = 0; i < number_clusters; i++)
    {
		for(int j = 0; j < dimension_data; j++)
        {
			double sum = 0;

            // Sum up every data row
			for(int k = i * number_sample; k < (i + 1) * number_sample; k++){
				sum += data[k][j];
			}
            // Calculate random centroid (just for initialization)
			centroid[i][j] = sum / number_sample;
		}
	}
}

/**
 * @brief Label the data for kmean calculation 
 * 
 * @param data N shapes data vector
 * @return     returns label 
 */
int Kmeans::Classify(std::vector<double> &data)
{
    int argmin;

	double min = -1;

	for(int j = 0; j < number_clusters; j++)
    {
		double distance = 0;
				
		for(int k = 0; k < dimension_data; k++)
        {
            // sum up the distance from the data and the centroid
			distance += (data[k] - centroid[j][k]) * (data[k] - centroid[j][k]);
		}
		distance = sqrt(distance);
		
        // if dinstance is greater than distance of datapoint before, set new label
		if(min == -1 || min > distance){
			argmin	= j;
			min		= distance;
		}
	}
	return argmin;
}

/**
 * @brief Calculate the mean of the labeled data matrix, also calculate the new centroid
 * 
 * @param number_data   N number of data
 * @param data          N x M shaped Matrix of data
 * @return              centroid movement
 */
double Kmeans::Cluster(int number_data, std::vector< std::vector<double> > data)
{
    double movements_centroids = 0;

	std::vector<int> label(number_data);
	std::vector<double> mean(dimension_data);

	for(int i = 0; i < number_data;i++)
    {
        // Label each data
		label[i] = Classify(data[i]);
	}

	for(int j = 0; j < number_clusters; j++)
    {
		int number_sample = 0;
		double movements = 0;

		for(int k = 0; k < dimension_data; k++)
        {
            // reset mean
			mean[k] = 0;
		}
		for(int i = 0;i < number_data;i++)
        {
			if(label[i] == j)
            {
				for(int k = 0; k < dimension_data; k++)
                {
                    // sum up the datapoints with the same label
					mean[k] += data[i][k];
				}
				number_sample++;
			}
		}
		for(int k = 0; k < dimension_data; k++)
        {
			if(number_sample) mean[k] /= number_sample;

			movements += (centroid[j][k] - mean[k]) * (centroid[j][k] - mean[k]);
            // set new centroid
			centroid[j][k] = mean[k];
		}
		movements_centroids += sqrt(movements);
	}

	return movements_centroids;
}

