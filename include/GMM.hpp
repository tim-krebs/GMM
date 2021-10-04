#pragma once

#include<string>
#include<vector>
#include<map>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <math.h>

#include "Kmeans.hpp"

struct Model
{
    std::vector<double> weight;
    std::vector<std::vector<double> > mean;
    std::vector<std::vector<double> > covariance;
    std::vector<std::vector<double> > invert_covariance;
    std::vector<double> ExpCoeff;
};

class GMM
{
private:
    /* data */
    Model newModel();
    void delModel(Model model);
    void completeModel(Model& model);
    double Likelihood(const std::vector<std::vector<double>>& melCepData, size_t frameCount, Model model, std::vector<std::vector<double>>& normProb, std::vector<double>& mixedProb);

    int m_MixDim;
    int m_MfccDim;
    double m_Threshold;
    double m_MinCov;
    Model m_Model;
    int number_gaussian_components;
    std::map<std::string, Model> m_Models;

    const double PI2 = 6.28318530717958647692;

public:
    GMM();
    virtual ~GMM();

    int Expectation_Maximation(const std::vector<std::vector<double> > melCepData, size_t frameCount);
    std::string Classify(const std::vector<std::vector<double> > &melCepData, size_t frameCount);
    double Likelihood(const std::vector<std::vector<double> > &melCepData, size_t frameCount);
    bool LoadModel(const std::string& filePath);
    bool SaveModel(const std::string& filePath);
    bool AddModel(const std::string& name);
    bool AddModel(const std::string& filePath, const std::string& name);
};

/**
 * @brief Construct a new GMM::GMM object
 * 
 */
GMM::GMM()
{
    // Set the mfcc dimension
    // Set the mixture dimensions
    m_MixDim = 12;
    m_MfccDim= 12;

    // Set break statement for testing
    m_Threshold = 0.005;
    m_MinCov = 0.015;

    // Create Models
    m_Model = newModel();
}

GMM::~GMM()
{
    delModel(m_Model);

    std::map<std::string, Model>::const_iterator it = m_Models.begin();
    while(it != m_Models.end())
    {
        delModel(it->second);
        ++it;
    }
    m_Models.clear();
}

/**
 * @brief Train the GMM with EM-Algorithm
 *        E: estimation step
 *        M: maximation step
 * 
 * @param melCepData (2d-vector) matrix contains the frames with features: melCepData(frames x 39)
 * @param frameCount (size_t) number of frames 
 * @return (int) number of training iterations
 */
int GMM::Expectation_Maximation(const std::vector<std::vector<double> > melCepData, size_t frameCount)
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

	//*** Initialization
	for(int i = 0; i < m_MixDim; i++)
    {
		m_Model.weight[i] = 1.0 / m_MixDim;
    }


    step = (int)floor(frameCount / m_MixDim);

	for(int j = 0; j < m_MixDim; j++)
    {
        for(int i = 0; i < m_MfccDim; i++)
        {
            m_Model.mean[i][j] = melCepData[step * (j + 1) -1][i];
        }
    }

	for(int i = 0; i < m_MixDim; i++)
    {
        for(int j = 0; j < m_MfccDim; j++)
        {
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
	    newProb = Likelihood(melCepData, frameCount, m_Model, normProb, mixedProb);

        // E process, a probability matrix, n*m_MixDim
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

	tempProb.clear();
	normProb.clear();
	sumProb.clear();
	mixedProb.clear();

    return iteration;
}

/**
 * @brief Decoder of the GMM
 * 
 * @param melCepData (2d-vector) matrix contains the frames with features: melCepData(frames x 39)
 * @param frameCount (size_t) number of frames 
 * @return (string) returns the recognized name
 */
std::string GMM::Classify(const std::vector<std::vector<double> >& melCepData, size_t frameCount)
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
double GMM::Likelihood(const std::vector<std::vector<double> >& melCepData, size_t frameCount)
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
 * @brief GMM model saver to text files
 * 
 * @param filePath (string) Filepath to save location
 * @return  true if the action was successful
 */
bool GMM::SaveModel(const std::string& filePath)
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
bool GMM::LoadModel(const std::string& filePath)
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
bool GMM::AddModel(const std::string& word)
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
bool GMM::AddModel(const std::string& filePath, const std::string& word)
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
 * @param model      (struct)   Struct of GMM Models
 * @param normProb   (double)   2D Matrix of NormalPrabability
 * @param mixedProb  (double)   Vector of mixed Probability
 * @return           (double)   Likelihood
 */
double GMM::Likelihood(const std::vector<std::vector<double> > &melCepData, size_t frameCount, Model model, std::vector<std::vector<double> > &normProb, std::vector<double> &mixedProb)
{
	double prob = 0.0;
    std::vector<double> maxMatrix(frameCount);
    std::vector<std::vector<double> > expMatrix(frameCount, std::vector<double>(m_MixDim, 0));


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

    // calculate Probability for each frame 
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
 * @return New GMM model with initial parameters
 */
Model GMM::newModel()
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
void GMM::delModel(Model model)
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
void GMM::completeModel(Model& model)
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
