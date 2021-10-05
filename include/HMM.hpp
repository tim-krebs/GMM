#pragma once

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <math.h>

#include "Matrix.hpp"

struct Model
{
    std::vector<double> weight;
    std::vector<std::vector<double> > mean;
    std::vector<std::vector<double> > covariance;
    std::vector<std::vector<double> > invert_covariance;
    std::vector<double> ExpCoeff;
};

class HMM
{
private:
    /* data */
    Model newModel();
    void delModel(Model model);
    void completeModel(Model& model);
    double Likelihood(const std::vector<std::vector<double>>& melCepData, size_t frameCount, Model model, std::vector<std::vector<double>>& normProb, std::vector<double>& mixedProb);

    int m_MixDim;
    int m_MfccDim;
    int num_states;
    double m_Threshold;
    double m_MinCov;
    Model m_Model;
    int number_gaussian_components;
    std::map<std::string, Model> m_Models;

    const double PI2 = 6.28318530717958647692;

    // Initital Probability pi
    std::vector<double> initial_probability;
    // Transition Probability A
    std::vector<std::vector<double> > state_transition_probability;
    // Observation Probability B
    std::vector<double> state_observation_probability;
    // Alpha
    std::vector<std::vector<double> > alpha;


public:
    HMM(int states, int mfcc_dim);
    virtual ~HMM();

    std::string Classify(const std::vector<std::vector<double> > &melCepData, size_t frameCount);
    double Likelihood(const std::vector<std::vector<double> > &melCepData, size_t frameCount);
    bool LoadModel(const std::string& filePath);
    bool SaveModel(const std::string& filePath);
    bool AddModel(const std::string& name);
    bool AddModel(const std::string& filePath, const std::string& name);

    double Gaussian_Distribution(std::vector<double> data, std::vector<double> &mean, std::vector<std::vector<double> > &covariance);
    int Expectation_Maximation(const std::vector<std::vector<double> > &melCepData, size_t frameCount);
    double Fordward_Algorithm(int num_states, std::vector<int> &state, std::vector<std::vector<double> > &state_transition_probability, std::vector<double> &state_observation_probability, std::vector<std::vector<double> > &alpha);
};

/**
 * @brief Construct a new HMM::HMM object
 * 
 * @param states    (int) Number of HMM states
 * @param mfcc_dim  (int) Dimension of MFCC Matrix
 */
HMM::HMM(int states, int mfcc_dim) 
{
    // Set the mfcc dimension
    // Set the mixture dimensions
    this->m_MixDim = mfcc_dim;
    this->m_MfccDim = mfcc_dim;
    this->num_states = states;

    // Initital Probability pi
    for(int i = 0; i < num_states; i++)
    {
        if(i == 0) initial_probability.push_back(1);
        initial_probability.push_back(0);
    }

    //Initialize the transition prob matrix A
    state_transition_probability.resize(num_states);
    for (int i = 0; i < num_states; i++) 
    {
        state_transition_probability[i].resize(num_states);
			for (int j = 0; j < num_states; j++) 
            {
                // for linear model
                if(i == 0 && j == 0)
                {
                    state_transition_probability[0][0] = 0.0;
                    state_transition_probability[0][1] = 1.0;
                }
				else if (i == j)
                {
                    state_transition_probability[i][j] = 0.7;
                    state_transition_probability[i][j+1] = 0.3;
                }
			}
		}
    state_transition_probability[num_states-1][num_states-1] = 0.0;

    // Create GMM Models
    m_Model = newModel();
}

HMM::~HMM()
{
    delModel(m_Model);

    std::map<std::string, Model>::const_iterator it = m_Models.begin();
    while(it != m_Models.end())
    {
        delModel(it->second);
        ++it;
    }
    //Delete models
        //m_Models.clear();
        //initial_probability.clear();
        //for(int i = 0; i < num_states; i++) 
        //{
        //    state_transition_probability[i].clear();
        //}
        //alpha.clear();
        //state_transition_probability.clear();
        //state_observation_probability.clear();
}


/**
 * @brief Train the HMM with EM-Algorithm
 *        E: estimation step
 *        M: maximation step
 * 
 * @param melCepData (2d-vector) matrix contains the frames with features: melCepData(frames x 39)
 * @param frameCount (size_t) number of frames 
 * @return (int) number of training iterations
 */
int HMM::Expectation_Maximation(const std::vector<std::vector<double> > &melCepData, size_t frameCount)
{
    int step;
	int iteration = 0;
	double newProb;
	double recentProb = 0.0;

    std::vector<double> sumProb(m_MixDim);
    std::vector<double> mixedProb(frameCount);
    std::vector<std::vector<double>> normProb(frameCount, std::vector<double>(m_MixDim));
    std::vector<std::vector<double>> tempProb(m_MfccDim, std::vector<double>(m_MixDim));
    std::vector<std::vector<double>> squareCep(frameCount, std::vector<double>(m_MfccDim));

	//*** Initialize the Gaussian mixture Models
	for(int i = 0; i < m_MixDim; i++)
    {
        // c_j
		m_Model.weight[i] = 1.0 / m_MixDim;
    }

    step = (int)floor(frameCount / m_MixDim);

	for(int j = 0; j < m_MixDim; j++)
    {
        for(int i = 0; i < m_MfccDim; i++)
        {
            // µ_jk
            m_Model.mean[i][j] = melCepData[step * (j + 1) -1][i];
        }
    }

	for(int i = 0; i < m_MixDim; i++)
    {
        for(int j = 0; j < m_MfccDim; j++)
        {
            // sigma_jk
            m_Model.covariance[i][j] = 1.0;
        }
    }

    for(size_t i = 0; i < frameCount; i++)
    {
        for(int j = 0; j < m_MfccDim; j++)
        {
            squareCep[i][j] = melCepData[i][j] * melCepData[i][j];
        }
    }

    // Iterative processing
    // EM-Algorithm
	while(true)
	{
	    completeModel(m_Model);
        // Computes the Expectation matrix and the Probability
	    newProb = Likelihood(melCepData, frameCount, m_Model, normProb, mixedProb);

        // E process
	   	for(size_t i = 0; i < frameCount; i++)
        {
		    for(int j = 0; j < m_MixDim; j++)
			{
               	normProb[i][j] *= m_Model.weight[j] / mixedProb[i];
            }
        }

        // M process
	    for(int i = 0; i < m_MixDim; i++)
		{
		   	sumProb[i] = 0.0;
		   	for(size_t j = 0; j < frameCount; j++)
            {
                sumProb[i] += normProb[j][i];
            }
		}

        // renew mixture coefficients
	   	for(int i = 0; i < m_MixDim; i++)
        {
            // c_j = Erwartungswert / anzahl datenpunkte
		   	m_Model.weight[i] = sumProb[i] / frameCount;
        }

        // renew mean
		for(int i = 0; i < m_MfccDim; i++)
        {
			for(int j = 0; j < m_MixDim; j++)
			{
				tempProb[i][j] = 0.0;
				for(size_t k = 0; k < frameCount; k++)
			    	tempProb[i][j] += melCepData[k][i] * normProb[k][j];

				m_Model.mean[i][j] = tempProb[i][j] / sumProb[j];
			}
        }

        // renew covariance
	   	for(int i = 0; i < m_MixDim; i++)
        {
		   	for(int j = 0; j < m_MfccDim; j++)
            {
                tempProb[j][i] = 0.0;
                for(size_t k = 0; k < frameCount; k++)
                {
                    tempProb[j][i] += squareCep[k][j] * normProb[k][i];
                }
                tempProb[j][i] = tempProb[j][i] / sumProb[i];
            }
        }

        // set covariance
	   	for(int i = 0; i < m_MixDim; i++)
        {
	     	for(int j = 0; j < m_MfccDim; j++)
			{
		   		m_Model.covariance[i][j] = tempProb[j][i] - m_Model.mean[j][i] * m_Model.mean[j][i];
			   	if(m_Model.covariance[i][j] <= m_MinCov)
                {
                    m_Model.covariance[i][j] = m_MinCov;
                }
			}
        }
        // prepare for next iteration
	  	recentProb = newProb;
        iteration++;
        if(iteration > 19)  break;
	}

    //Set the variables for the gaussian mixture Model
    state_observation_probability.resize(m_MixDim);
    for(int i = 0; i < m_MixDim; i++)
    {
        state_observation_probability[i] = m_Model.weight[i] * Gaussian_Distribution(melCepData[i], m_Model.mean[i], m_Model.covariance);
    }

	tempProb.clear();
	normProb.clear();
	sumProb.clear();
	mixedProb.clear();

    return iteration;
}


double HMM::Fordward_Algorithm(int num_states, std::vector<int> &state, std::vector<std::vector<double> > &state_transition_probability, std::vector<double> &state_observation_probability, std::vector<std::vector<double> > &alpha)
{
    double log_likelihood = 0;

    alpha.resize(state_observation_probability.size());
    for(int i = 0; i < state_observation_probability.size(); i++)
    {
        alpha[i].resize(state_observation_probability.size());
    }


    for(int t = 0; t < num_states; t++)
    {
        double tmp = 0;

        if(t == 0)
        {
            //Set the first probability
            for(int i = 0; i < num_states; i++)
            {
                int j = state[i];
                alpha[t][i] = initial_probability[i] * state_observation_probability[j];
            }
        }
        else
        {
            for(int i = 0; i < num_states; i++)
            {
                double sum = 0;
                for(int j = 0; j < num_states-1; j++)
                {
                    // Compute other forward-probabilities
                    int k = state[i];
					int l = state[j];
                    sum += alpha[t - 1][j] * state_transition_probability[l][k];

                }
            }
        }
    }
    return -log_likelihood;
}

/**
 * @brief Decoder of the HMM
 * 
 * @param melCepData (2d-vector) matrix contains the frames with features: melCepData(frames x 39)
 * @param frameCount (size_t) number of frames 
 * @return (string) returns the recognized name
 */
std::string HMM::Classify(const std::vector<std::vector<double> >& melCepData, size_t frameCount)
{
    double likelihood;
    double probMax = 0;
    std::string name;
    bool first = true;

    std::vector<double> mixedProb;
    std::vector<std::vector<double> > normProb;

    std::map<std::string, Model>::const_iterator it = m_Models.begin();
    std::map<std::string, Model>::const_iterator itEnd = m_Models.end();

    mixedProb.resize(frameCount);
    normProb.resize(frameCount);

    for(size_t i = 0; i < frameCount; i++)
    {
        normProb[i].resize(m_MixDim);
    }

    while(it != itEnd)
    {
        likelihood = Likelihood(melCepData, frameCount, it->second, normProb, mixedProb);

        if((first == true) || (probMax <= likelihood))
        {
            probMax = likelihood;
            name = it->first;
            first = false;
        }
        ++it;
    }

    normProb.clear();
	mixedProb.clear();

    return name;
}

/**
 * @brief Calculates the Probability for each frame
 * 
 * @param melCepData (2d-vector) matrix contains the frames with features: melCepData(frames x 39)
 * @param frameCount (size_t) number of frames 
 * @return (double) returns probability  
 */
double HMM::Likelihood(const std::vector<std::vector<double> >& melCepData, size_t frameCount)
{
    double prob;
    std::vector<double> mixedProb;
    std::vector<std::vector<double> > normProb;

    mixedProb.resize(frameCount);
    normProb.resize(frameCount);

    for(size_t i = 0; i<frameCount; i++)
    {
        normProb[i].resize(m_MixDim);
    }

    prob = Likelihood(melCepData, frameCount, m_Model, normProb, mixedProb);

    normProb.clear();
	mixedProb.clear();

    return prob;
}

/**
 * @brief HMM model saver to text files
 * 
 * @param filePath (string) Filepath to save location
 * @return  true if the action was successful
 */
bool HMM::SaveModel(const std::string& filePath)
{
    std::ofstream outFile(filePath);
    if(!outFile.is_open())
    {
        return false;
    }


    outFile << std::fixed << std::setprecision(6);
    outFile << "mixcoef:" << std::endl;

    for(int i = 0; i < m_MixDim; i++)
    {
        outFile << m_Model.weight[i] << " ";
    }


    outFile << std::endl << "mean:" << std::endl;

    for(int i = 0; i < m_MfccDim; i++)
    {
        for(int j = 0; j < m_MixDim; j++)
        {
           outFile << m_Model.mean[i][j] << " ";
        }
 
        outFile << std::endl;
    }

    outFile << "covariance:" << std::endl;

    for(int i = 0; i < m_MixDim; i++)
    {
        for(int j = 0; j < m_MfccDim; j++)
        {
            outFile << m_Model.covariance[i][j] << " ";
        }

        outFile << std::endl;
    }

    outFile.close();
    return true;
}

/**
 * @brief Model loader from save location
 * 
 * @param filePath (string) File path to saved location
 * @return  true if the action was successful
 */
bool HMM::LoadModel(const std::string& filePath)
{
    std::string title;

    std::ifstream inFile(filePath, std::ifstream::in);
    if(!inFile.is_open())
    {
        return false;
    }

    inFile >> title;
    for(int i = 0; i < m_MixDim; i++)
    {
        inFile >> m_Model.weight[i];
    }

    inFile >> title;
    for(int i = 0; i < m_MfccDim; i++)
    {
        for(int j = 0; j < m_MixDim; j++)
        {
            inFile >> m_Model.mean[i][j];
        }
    }

    inFile >> title;
    for(int i=0; i<m_MixDim; i++)
    {
        for(int j=0; j<m_MfccDim; j++)
        {
            inFile >> m_Model.covariance[i][j];
        }
    }

    inFile.close();
    completeModel(m_Model);
    return true;
}

/**
 * @brief Create new statistical Model with initial parameters
 * 
 * @param word (string) 
 * @return
 */
bool HMM::AddModel(const std::string& word)
{
    // Create new Model
    Model model;
    model = newModel();

    for(int i = 0; i < m_MixDim; i++)
    {
        model.weight[i] = m_Model.weight[i];
        model.ExpCoeff[i] = m_Model.ExpCoeff[i];
    }

    for(int i = 0; i < m_MfccDim; i++)
    {
        for(int j = 0; j < m_MixDim; j++)
        {
            model.mean[i][j] = m_Model.mean[i][j];
            model.covariance[j][i] = m_Model.covariance[j][i];
            model.invert_covariance[j][i] = m_Model.invert_covariance[j][i];
        }
    }

    m_Models[word] = model;

    return true;
}

/**
 * @brief 
 * 
 * @param filePath (string) Filepath to load location
 * @param word     (string) 
 * @return
 */
bool HMM::AddModel(const std::string& filePath, const std::string& word)
{
    if(!LoadModel(filePath))
    {
        return false;
    }

    // Add Model to modelset
    AddModel(word);

    return true;
}

/**
 * @brief Computes the Likelihoof for each frame
 * 
 * @param melCepData (double)   2D Matrix of MFCC data
 * @param frameCount (size_t)   Number of frames
 * @param model      (struct)   Struct of HMM Models
 * @param normProb   (double)   2D Matrix of NormalPrabability
 * @param mixedProb  (double)   Vector of mixed Probability
 * @return           (double)   Likelihood
 */
double HMM::Likelihood(const std::vector<std::vector<double> > &melCepData, size_t frameCount, Model model, std::vector<std::vector<double> > &normProb, std::vector<double> &mixedProb)
{
	double prob = 0.0;
    std::vector<double> maxMatrix(frameCount);
    std::vector<std::vector<double> > expMatrix(frameCount, std::vector<double>(m_MixDim, 0));

    // Calculate the Expectation Matrix
    for(size_t i = 0; i < frameCount; i++ )
    {
        for(int j = 0; j < m_MixDim; j++)
        {
            for(int k = 0; k < m_MfccDim; k++)
            {
                expMatrix[i][j] += (pow((melCepData[i][k] - model.mean[k][j]),2 )) * model.invert_covariance[j][k];
            }
        }
    }

    for(size_t i = 0; i < frameCount; i++)
    {
        auto it = max_element(expMatrix[i].begin(), expMatrix[i].end());
        maxMatrix[i] = *it;
    }

    // calculate Probability for each frame and expectaion matrix 
    for(size_t i = 0; i < frameCount; i++)
    {
        mixedProb[i] = 0.0;
        for(int j = 0; j < m_MixDim; j++)
        {
            expMatrix[i][j] = exp(expMatrix[i][j] - maxMatrix[i]);
            normProb[i][j] = expMatrix[i][j] * model.ExpCoeff[j];
            mixedProb[i] = mixedProb[i] + normProb[i][j] * model.weight[j];
        }
        prob += log(mixedProb[i]) + maxMatrix[i];
    }

	expMatrix.clear();
	maxMatrix.clear();

    return prob;
}

/**
 * @brief Creates new Initial GMM Model
 * 
 * @return New HMM model with initial parameters
 */
Model HMM::newModel()
{
    Model model;

    model.weight.resize(m_MixDim);
    model.mean.resize(m_MfccDim);

    for(int i = 0; i < m_MfccDim; i++)
    {
        model.mean[i].resize(m_MixDim);
    }

    // Resize matrix to m_MixDim (number of mixtures)
    model.covariance.resize(m_MixDim);
    model.invert_covariance.resize(m_MixDim);

    for(int i = 0; i < m_MixDim; i++)
    {
        model.covariance[i].resize(m_MfccDim);
        model.invert_covariance[i].resize(m_MfccDim);
    }

    model.ExpCoeff.resize(m_MixDim);

    return model;
}

/**
 * @brief Deletes allcreated models
 * 
 * @param model (struct) struct of models
 */
void HMM::delModel(Model model)
{
    model.weight.clear();
    model.mean.clear();
    model.covariance.clear();
    model.invert_covariance.clear();
    model.ExpCoeff.clear();
}

/**
 * @brief 
 * 
 * @param model 
 */
void HMM::completeModel(Model& model)
{
    double x = pow(PI2, (-m_MfccDim / 2));

    for(int i = 0; i < m_MixDim; i++)
    {
        model.ExpCoeff[i] = 1.0;

        for(int j = 0; j < m_MfccDim; j++)
        {
            model.ExpCoeff[i] *= 1.0 / model.covariance[i][j];
        }

        model.ExpCoeff[i] = x * sqrt(model.ExpCoeff[i]);
    }

    for(int i = 0; i < m_MixDim; i++)
    {
        for(int j = 0; j < m_MfccDim; j++)
        {
            model.invert_covariance[i][j] = (-0.5) / model.covariance[i][j];
        }
    }
}


/**
 * @brief Computes the gaussian distrubution for the state observation
 * 
 * @param data          (double)    Vector of data
 * @param mean          (double)    Vector of means
 * @param covariance    (double) Matrix of covariances
 * @return  Observationprobability
 */
double HMM::Gaussian_Distribution(std::vector<double> data, std::vector<double> &mean, std::vector<std::vector<double> > &covariance)
{
    double result;
	double sum = 0;

	std::vector<std::vector<double> >inversed_covariance;

	Matrix matrix;

    inversed_covariance.resize(data.size());
	for (int i = 0; i < data.size(); i++){
		inversed_covariance[i].resize(data.size());
	}
	matrix.Inverse("diogonal", data.size(), covariance, inversed_covariance);

	for (int i = 0; i < data.size(); i++){
		double partial_sum = 0;

		for (int j = 0; j < data.size(); j++){
			partial_sum += (data[j] - mean[j]) * inversed_covariance[j][i];
		}
		sum += partial_sum * (data[i] - mean[i]);
	}

	for (int i = 0; i < data.size(); i++){
		inversed_covariance[i].clear();
	}
	inversed_covariance.clear();

	result = 1.0 / (pow(2 * 3.1415926535897931, data.size() / 2.0) * sqrt(matrix.Determinant("diagonal", data.size(), covariance))) * exp(-0.5 * sum);

	return result;
}



