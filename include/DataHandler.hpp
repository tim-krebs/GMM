#pragma once

#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string>
#include <map>

#define SECONDS 3
#define FREQ    16000              //You can try 48000 to use 48000Hz wav files, but it's more slow.
#define TRAINSIZE FREQ * SECONDS        //4 secondes of voice for trainning
                                // --- you can increase this value to improve the recognition rate
#define RECOGSIZE FREQ * SECONDS        //1 seconde of voice for recognition
                                // --- you can increase this value to improve the recognition rate

class DataHandler{

private:


public:
    //Constructor
    DataHandler()
    {
        
    }

    // Destructor
    ~DataHandler()
    {

    }

    std::string GetWord(int wordId);
    size_t ReadWav(const std::string& filePath, short int voiceData[], size_t sizeData, size_t seek);
    std::string GetFilePath(int person, int num, int mode, const std::string& extention);
    void CheckWavHeader(char *header);
    /**
     * @brief This function reads a .txt-file, extract the data and push it into a Vector
     * @param stereo    false if .wav return vector should monp
     * @param path      Path to the .txt-file which should be read in
     * @return          returns the signal vector which contains all the data
     */
    std::vector<double> read_speech_data(bool stereo, std::string &path)
    {
        // open Speech txt file
        std::string str;
        std::vector<double> signal;
        std::vector<double> mono;
        std::ifstream file;
        file.open(path);
        if(file.is_open())
        {
            while(std::getline(file, str)){
                signal.push_back(double(std::stod(str)));
            }
        }
        file.close();


        if(!stereo)
        {
            for(int i = 0; i < signal.size(); i+=2)
            {
                double t = signal[i];
                double t_1  = signal[i+1];
                mono.push_back((t/2) + (t_1/2));
            }
        }
        else
        {
            return signal;
        }
        return mono;
    }

    /**
     * @brief This Function saves the modified signal in an .txt-file
     * 
     * @param path      Path where the .txt-file should be stored
     */
    void write_speech_data(std::string &path, std::vector<double> signal)
    {
        //Write to txt file
        std::ofstream offile;
        std::string strr;
        double tmp;


        offile.open(path);

        if(offile.is_open()){
            std::cout << "File ist open" << std::endl;
            for(int i = 0; i < signal.size(); i++)
            {
                offile << signal[i] << '\n';
            }
        }
        offile.close();
        std::cout << "Signal exported successfully!" << std::endl;;
    }
};


/**
 * @brief Set wordId. Is necessary for modelling the gmm 
 * 
 * @param wordId (int) Word keyword
 * @return (string) 
 */
std::string DataHandler::GetWord(int wordId)
{
    std::ostringstream oss;

    oss << "Word" << wordId;
    return oss.str();
}

/**
 * @brief This function handles the gmm model files, train wav files and the recognition wav files
 * 
 * @param word (int) Key which word is read in 
 * @param num  (int) number of files for each word
 * @param mode (int) decider for train or recognition step
 * @param extention (string) important for wav files
 * @return (string) datapath to wav file
 */
std::string DataHandler::GetFilePath(int word, int num, int mode, const std::string& extention)
{
    std::ostringstream oss;

    switch(mode)
    {
        case 0 :
            oss << "train/F0" << word << "_" << num << "-" << FREQ << "." << extention;
            break;
        case 1 :
            oss << "recog/F0" << word << "-" << FREQ << "." << extention;
            break;
        case 2 :
            oss << "model/" << word << "_" << num << "." << extention;
            break;
    }

    return oss.str();
}

/**
 * @brief Checks the specification of the wav file
 *        (Number of bits, Samplefrequency ..)
 * 
 * @param header (char)
 */
void DataHandler::CheckWavHeader(char *header)
{
	int sr;

	if (header[20] != 0x1)
		std::cout << std::endl << "Input audio file has compression [" << header[20] << "] and not required PCM" << std::endl;

	sr = ((header[24] & 0xFF) | ((header[25] & 0xFF) << 8) | ((header[26] & 0xFF) << 16) | ((header[27] & 0xFF) << 24));
    std::cout << " " << (int)header[34] << " bits, " << (int)header[22] << " channels, " << sr << " Hz";
}

/**
 * @brief .Wav file reader 
 * 
 * @param filePath (string) filepath to wav file
 * @param voiceData (int) Array container for speech data
 * @param sizeData (size_t) size of data
 * @param seek (size_t) default = 0 for wav file
 * @return  (siez_t)
 */
size_t DataHandler::ReadWav(const std::string& filePath, short int voiceData[], size_t sizeData, size_t seek)
{
    std::ifstream inFile(filePath, std::ifstream::in|std::ifstream::binary);
    size_t ret;

    if(!inFile.is_open())
    {
        std::cout << std::endl << "Can not open the WAV file !!" << std::endl;
        return -1;
    }

    char waveheader[44];
    inFile.read(waveheader, 44);
    if(seek==0) CheckWavHeader(waveheader);

    if(seek!=0) inFile.seekg (seek*sizeof(short int), std::ifstream::cur);

    inFile.read(reinterpret_cast<char *>(voiceData), sizeof(short int)*sizeData);
    ret = (size_t)inFile.gcount()/sizeof(short int);

    inFile.close();
    return ret;
}

