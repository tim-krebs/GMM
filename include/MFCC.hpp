#pragma once

#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <math.h>

class MFCC
{
private:
    /* data */

    void fft(std::vector<double>& data, int nn, int isign);
    void setFilterBank();
    void setDCTCoeff();
    void setLiftCoeff();
    void Analyse(std::vector<std::vector<double>>& postData, size_t frameCount, size_t currrentFrame);

    double freq2mel(double freq);
    double mel2freq(double mel);

    //Settings
    int m_Frequence;
    int m_FrameSize;
    int m_FrameShift;
    int m_FilterNumber;
    int m_MFCCDim;

    //Internal
    size_t m_FrameCount;
    std::vector<double> m_WindowCoefs;
    std::vector<std::vector<double>> m_FilterBank;
    std::vector<std::vector<double>> m_DCTCoeff;
    std::vector<double> m_CepLifter;
    std::vector<std::vector<double>> m_MFCCData;

    size_t m_CurrentFrame;
    std::vector<short int> m_RestData;

    //Constantes
    static const double PI;
    static const double PI2;
    static const double PI4;
public:
    enum WindowMethod
    {
        Hamming, 
        Hann, 
        Blackman, 
        None
    };
    MFCC();
    MFCC(int freq, int size, int shift, WindowMethod method, int filterNum, int MFCCcDim);
    virtual ~MFCC();

    size_t Analyse(const short int data[], size_t sizeData);
    bool Save(const std::string& filePath);
    const std::vector<std::vector<double>>& GetMFCCData();

    void setWindowMethod(WindowMethod method);
    void StartAnalyse(size_t maxSize);
    bool AddBuffer(const short int data[], size_t sizeData);
    size_t GetFrameCount();

};

/**
 * @brief Construct a new MFCC::MFCC object
 * 
 */
MFCC::MFCC() : MFCC(16000, 16, 8, WindowMethod::Hamming, 24, 12)
{
}

/**
 * @brief Construct a new MFCC::MFCC object
 * 
 * @param freq      (int)   Frequency of the Signal
 * @param size      (int)   Frame length, default is 0.025
 * @param shift     (int)   Frame shift, default is 0.010
 * @param method    (enum)  Window type: Hamming, Hann, Blackman, None
 * @param filterNum (int)   Number of filters for the Mel-filterbank
 * @param MFCCDim   (int)   Dimension of the MFCC matrix
 */
MFCC::MFCC(int freq, int size, int shift, WindowMethod method, int filterNum, int MFCCDim)
{
    m_Frequence = freq;
    m_FrameSize = freq*size/1000;
    m_FrameShift= freq*shift/1000;
    m_FilterNumber=filterNum;
    m_MFCCDim   = MFCCDim;

    m_FrameCount  = 0;
    m_CurrentFrame= 0;

    setWindowMethod(method);
    setFilterBank();
    setDCTCoeff();
    setLiftCoeff();
}

/**
 * @brief Destroy the MFCC::MFCC object
 * 
 */
MFCC::~MFCC()
{
    m_WindowCoefs.clear();
    m_FilterBank.clear();
    m_DCTCoeff.clear();
    m_CepLifter.clear();
    m_MFCCData.clear();
    m_RestData.clear();
}

/**
 * @brief 
 * 
 * @param data      (short int) data vector
 * @param sizeData  (size_t)    Lenght of data vector
 * @return          Frame count of MFCC data
 */
size_t MFCC::Analyse(const short int data[], size_t sizeData)
{
    std::vector<std::vector<double> > postData;

    ///*** Initialisation
    m_MFCCData.clear();
    m_CurrentFrame = 0;
    m_FrameCount = (sizeData - m_FrameSize + m_FrameShift) / m_FrameShift;
    m_MFCCData.resize(m_FrameCount);
    postData.resize(m_FrameCount);

    ///*** Apply the window coefficients
    for(size_t i = 0; i < m_FrameCount; i++)
    {
        for(int j = 0; j < m_FrameSize; j++)
        {
            // Appply Hamming window to function
            postData[i].push_back(data[i * m_FrameShift + j] * m_WindowCoefs[j]);
            postData[i].push_back(0);
        }
    }

    ///*** Analyse (FFT, E, Filterbank)
    Analyse(postData, m_FrameCount, 0);
    postData.clear();

    return m_FrameCount;
}

/**
 * @brief 
 * 
 * @param postData       (double)   Data vector
 * @param frameCount     (size_t)   Number of MFCC frames 
 * @param currrentFrame  (size_t)   Number of current MFCC frames
 */
void MFCC::Analyse(std::vector<std::vector<double> >& postData, size_t frameCount, size_t currrentFrame)
{
    std::vector<std::vector<double>> spectralPower;
    std::vector<std::vector<double>> melSpectralPower;


    ///*** FFT matrix
    for(size_t i = 0; i < frameCount; i++)
    {
        fft(postData[i], m_FrameSize, 1);
    }

    ///*** Energy matrix
    spectralPower.resize(frameCount);

    for(size_t i = 0; i < frameCount; i++)
    {
        for(int j = 0; j < m_FrameSize / 2+1; j++)
        {
            spectralPower[i].push_back(postData[i][j<<1] * postData[i][j<<1] + postData[i][(j<<1) + 1] * postData[i][(j<<1) +1 ]);
        }
    }



    melSpectralPower.resize(m_FilterNumber);

    ///*** Apply filter bank
    for(int i = 0; i < m_FilterNumber; i++)
    {
        for(size_t k = 0; k < frameCount; k++)
        {
            melSpectralPower[i].push_back(0);
            for(int j = 0; j < m_FrameSize / 2+1; j++)
            {
                melSpectralPower[i][k] += m_FilterBank[i][j] * spectralPower[k][j];
            }

            melSpectralPower[i][k] = log(melSpectralPower[i][k]);
        }
    }

    spectralPower.clear();

    ///*** MFCCc matrix
    for(size_t k = 0; k < frameCount; k++)
    {
        //MFCC + e
        for(int i = 0; i < (m_MFCCDim); i++)
        {
            m_MFCCData[currrentFrame+k].push_back(0);
            for(int j = 0; j < m_FilterNumber; j++)
            {
                m_MFCCData[currrentFrame+k][i] += m_DCTCoeff[i][j] * melSpectralPower[j][k];
            }
        }
    }

    melSpectralPower.clear();

    ///*** Ceplift
    for(size_t i = 0; i < frameCount; i++)
    {
        for(int j = 0; j < m_MFCCDim; j++)
        {
            m_MFCCData[currrentFrame+i][j] *= m_CepLifter[j];
        }
    }

    return;
}

/**
 * @brief Saves the MFCC extracted data to file
 * 
 * @param filePath (string) Path where the MFCC get saved
 * @return         (bool) If operation was successful
 */
bool MFCC::Save(const std::string& filePath)
{
	size_t frameCount;
    std::ofstream outFile(filePath);

    if(!outFile.is_open())
        return false;

    outFile << std::fixed << std::setprecision(6);

    if(m_CurrentFrame > 0 )
        frameCount = m_CurrentFrame;
    else
        frameCount = m_FrameCount;

    for(size_t i=0; i<frameCount; i++)
    {
        for(int j=0; j<m_MFCCDim; j++)
            outFile << m_MFCCData[i][j] << " ";
        outFile << std::endl;
    }

    outFile.close();
    return true;
}

/**
 * @brief Returns the MFCC data vector
 * 
 * @return 2 D Vector with MFCC data
 */
const std::vector<std::vector<double> >& MFCC::GetMFCCData()
{
    return m_MFCCData;
}

void MFCC::StartAnalyse(size_t maxSize)
{
    m_CurrentFrame = 0;
    m_FrameCount = (maxSize-m_FrameSize+m_FrameShift)/m_FrameShift;
    m_MFCCData.clear();
    m_MFCCData.resize(m_FrameCount);

    m_RestData.clear();
}

bool MFCC::AddBuffer(const short int data[], size_t sizeData)
{
    std::vector<std::vector<double> > postData;
    size_t restSize = m_RestData.size();
    size_t frameCount;
	size_t i, k;
    int j;


    ///*** Initialisation
    if(m_FrameCount==0) return false;
    frameCount = (sizeData+restSize-m_FrameSize+m_FrameShift)/m_FrameShift;
    if(m_CurrentFrame+frameCount>m_FrameCount)
    {
        frameCount = m_FrameCount-m_CurrentFrame;
        if(frameCount == 0) return false;
    }
    postData.resize(frameCount);

    ///*** Apply the window coefficients
    for(i = 0; i < frameCount; i++)
    {
        for(j=0; j<m_FrameSize; j++)
        {
            k = i*m_FrameShift+j;
            if(k<restSize)
                postData[i].push_back(m_RestData[k]*m_WindowCoefs[j]);
            else
                postData[i].push_back(data[i*m_FrameShift+j-restSize]*m_WindowCoefs[j]);

            postData[i].push_back(0);
        }
    }
    m_RestData.clear();

    ///*** Analyse
    Analyse(postData, frameCount, m_CurrentFrame);
    postData.clear();
    m_CurrentFrame += frameCount;
    if(m_CurrentFrame>=m_FrameCount) return false;

    ///*** Memorize the remainder
    restSize = (sizeData+restSize)-frameCount*m_FrameShift;
    if(restSize>0)
    {
        for(i=sizeData-restSize; i<sizeData; i++)
            m_RestData.push_back(data[i]);
    }

    return true;
}

size_t MFCC::GetFrameCount()
{
    return m_CurrentFrame;
}

/**
 * @brief Computes the FFT of a dataframe 
 * 
 * @param data  (double)
 * @param nn    (int)
 * @param isign (int)
 */
void MFCC::fft(std::vector<double>& data, int nn, int isign)
{
	int i,j,m,n,mmax,istep;
	double wtemp,wr,wpr,wpi,wi,theta;
	double tempr,tempi;


	n=nn<<1;
	j=1;
	for(i=1;i<n;i+=2)
	{
		if(j>i)
		{
		    std::swap(data[j-1], data[i-1]);
			std::swap(data[j], data[i]);
		}
		m=nn;

		while(m>=2&&j>m)
		{
            j-=m;
			m>>=1;
        }
		j+=m;
	}

	mmax=2;
	while(n>mmax)
	{
		istep=mmax<<1;
		theta=isign*(PI2/mmax);
		wtemp=sin(0.5*theta);
		wpr=-2.0*wtemp*wtemp;
		wpi=sin(theta);
		wr=1.0;
		wi=0.0;
		for(m=1;m<mmax;m+=2)
		{
			for(i=m;i<=n;i+=istep)
			{
			    if(i>=n) continue;      //Buffer overflow if nn is not a power of 2
   			    j=i+mmax;
   			    if(j>=n) continue;      //Buffer overflow if nn is not a power of 2
			    tempr=wr*data[j-1]-wi*data[j];
			    tempi=wr*data[j]+wi*data[j-1];
                data[j-1]=data[i-1]-tempr;
			    data[j]=data[i]-tempi;
			    data[i-1]+=tempr;
			    data[i]+=tempi;
			}
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
		}
		mmax=istep;
	}
}

/**
 * @brief Implementation of the different Windows
 * 
 * @param method (enum)
 */
void MFCC::setWindowMethod(WindowMethod method)
{
    m_WindowCoefs.clear();

    switch(method)
    {
        case WindowMethod::Hamming :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(0.54-0.46*(cos(PI2*(double)i/(m_FrameSize))));
            break;

        case WindowMethod::Hann :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(0.5-0.5*(cos(PI2*(double)i/(m_FrameSize-1))));
            break;

        case WindowMethod::Blackman :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(0.42-0.5*(cos(PI2*(double(i)/(m_FrameSize-1))))+0.08*(cos(PI4*(double(i)/(m_FrameSize-1)))));
            break;

        case WindowMethod::None :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(1);
            break;
    }
}

/**
 * @brief Converts Frequency to Mel 
 * 
 * @param freq (double)
 * @return     (double)
 */
double MFCC::freq2mel(double freq)
{
	return 1125 * log10(1 + freq / 700);
}

/**
 * @brief Converts Mel to Frequency
 * 
 * @param mel (double)
 * @return    (double)
 */
double MFCC::mel2freq(double mel)
{
	return (pow(10, mel / 1125) - 1) * 700;
}

/**
 * @brief Computes Filterbank
 * 
 */
void MFCC::setFilterBank()
{
	double maxMel, deltaMel;
	double lowFreq, mediumFreq, highFreq, currentFreq;
	int filterSize = m_FrameSize/2+1;

	maxMel = freq2mel(m_Frequence/4);
	deltaMel = maxMel / (m_FilterNumber + 1);

	m_FilterBank.resize(m_FilterNumber);
    lowFreq = mel2freq(0);
    mediumFreq = mel2freq(deltaMel);
	for(int i = 0; i < m_FilterNumber; i++)
    {
		highFreq = mel2freq(deltaMel*(i+2));

		for(int j = 0; j < filterSize; j++)
		{
			currentFreq = (j*1.0 / (filterSize - 1) * (m_Frequence / 4));

			if((currentFreq >= lowFreq)&&(currentFreq <= mediumFreq))
				m_FilterBank[i].push_back(2*(currentFreq - lowFreq) / (mediumFreq - lowFreq));
			else if((currentFreq >= mediumFreq)&&(currentFreq <= highFreq))
				m_FilterBank[i].push_back(2*(highFreq - currentFreq) / (highFreq - mediumFreq));
			else
				m_FilterBank[i].push_back(0);
		}

		lowFreq = mediumFreq;
		mediumFreq = highFreq;
	}
}

/**
 * @brief Computes DCT Coefficients
 * 
 */
void MFCC::setDCTCoeff()
{
    m_DCTCoeff.resize(m_MFCCDim);
	for(int i = 0; i < m_MFCCDim; i++)
		for(int j = 0; j < m_FilterNumber; j++)
			m_DCTCoeff[i].push_back(2*cos((PI*(i+1)*(2*j + 1)) / (2 * m_FilterNumber)));
}

/**
 * @brief Computes Lifter Coefficients
 * 
 */
void MFCC::setLiftCoeff()
{
	for(int i = 0; i < m_MFCCDim; i++)
        m_CepLifter.push_back((1.0+0.5*m_MFCCDim*sin(PI*(i+1)/(m_MFCCDim)))/((double)1.0+0.5*m_MFCCDim));
}

