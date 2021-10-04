#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "MFCC.hpp"
#include "GMM.hpp"
#include "DataHandler.hpp"

#define NUM_WORDS   16

// private typedefs
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds Milliseconds;

// Private variables
const double MFCC::PI  = 3.14159265358979323846;
const double MFCC::PI2 = 2*PI;
const double MFCC::PI4 = 4*PI;

// Private prototypes
std::string GetWord(std::string name);


int main()
{
    // Create constructor
   	Clock::time_point trainStart, trainEnd, recogScoreStart, recogScoreEnd, recogPercentStart, recogPercentEnd;
    Milliseconds ms;
    DataHandler datahandler;
    GMM gmm;

    // Declare variables
	std::string filePath, name;
    short int bigVoiceBuffer[TRAINSIZE], littleVoiceBuffer[2000];
	size_t frameCount, realSize;
    std::vector<std::vector<double> > melCepData;
    int replacements = 0, omissions= 0, insertions = 0, wrong_word = 0, loop;
    std::map<std::string, int> recognizer;

    // Initialize MFCC
    MFCC mfcc(16000, 25, 10, MFCC::Hamming, 40, 12);

    /***************************************************************************
     * TRAINNING *
    ***************************************************************************/
    std::cout << "*** TRAINNING ***" << std::endl;
    trainStart = Clock::now();

    for(int wordId = 0; wordId <= NUM_WORDS; wordId++)
    {
        for(int num = 1; num <= 3; num++)
        {
            //** Load wav file
            std::string path = "/Users/timkrebs/OneDrive/Uni/8.Semester/Bachelorarbeit/02_Programme/C++/ASR_GMM/";
            filePath = datahandler.GetFilePath(wordId, num, 0, "wav");
            filePath = path.append(filePath);

            realSize = datahandler.ReadWav(filePath, bigVoiceBuffer, TRAINSIZE, 0);
            // Check if read operation was successfull
            if(realSize < 1) continue;


            //** Mfcc analyse WITH BIG BUFFER
            frameCount = mfcc.Analyse(bigVoiceBuffer,realSize);
            melCepData = mfcc.GetMFCCData();

            filePath.erase();
            path = "/Users/timkrebs/OneDrive/Uni/8.Semester/Bachelorarbeit/02_Programme/C++/ASR_GMM/";

            //** GMM trainning
            loop = gmm.Expectation_Maximation(melCepData, frameCount);
            filePath = datahandler.GetFilePath(wordId, num, 2, "gmm");

            filePath = path.append(filePath);

            gmm.SaveModel(filePath);

            std::cout << " : " << loop << " trainning loops" << std::endl;
            filePath.erase();
        }

    }
    trainEnd = Clock::now();

    //** Reload saved models for Recognition task
    for(int wordId = 0; wordId <= NUM_WORDS; wordId++)
    {
        for(int num=1; num<=3; num++)
        {
            std::string path = "/Users/timkrebs/OneDrive/Uni/8.Semester/Bachelorarbeit/02_Programme/C++/ASR_GMM/";
            filePath = datahandler.GetFilePath(wordId, num, 2, "gmm");
            filePath = path.append(filePath);

            gmm.AddModel(filePath, datahandler.GetWord(wordId));

            filePath.erase();
        }
    }

    /***************************************************************************
     * RECOGNITION best score *
    ***************************************************************************/
    std::cout << std::endl << "*** RECOGNITION best score ***" << std::endl;
    recogScoreStart = Clock::now();

    for(int wordId = 0; wordId <= NUM_WORDS; wordId++)
    {
        std::string path = "/Users/timkrebs/OneDrive/Uni/8.Semester/Bachelorarbeit/02_Programme/C++/ASR_GMM/";

        //** Mfcc analyse
        filePath = datahandler.GetFilePath(wordId, 0, 1, "wav");
        filePath = path.append(filePath);

        realSize = datahandler.ReadWav(filePath, bigVoiceBuffer, RECOGSIZE, 0);
        if(realSize < 1) continue;

        frameCount = mfcc.Analyse(bigVoiceBuffer,realSize);
        melCepData = mfcc.GetMFCCData();

        // Returns Modelname with the best probability
        name = gmm.Classify(melCepData, frameCount);

        // Get the name from the Codebook
        if(name == datahandler.GetWord(wordId))
        {
            name = GetWord(name);
            std::cout << " recognize correctly : " << name << std::endl;
        }
        else
        {
            name = GetWord(name);
            std::cout << " recognize wrong: " << name <<  std::endl;
        }
        filePath.erase();
    }
    recogScoreEnd = Clock::now();
    std::cout << std::endl;


    /***************************************************************************
     * RECOGNITION in % *
    ***************************************************************************/
    std::cout << std::endl << "*** RECOGNITION best percentage ***" << std::endl;
    recogPercentStart = Clock::now();

    for(int wordId = 0; wordId <= NUM_WORDS; wordId++)
    {
        std::string path = "/Users/timkrebs/OneDrive/Uni/8.Semester/Bachelorarbeit/02_Programme/C++/ASR_GMM/";

        //** Mfcc analyse 
        bool bcontinue = true;
        size_t position = 0;
        mfcc.StartAnalyse(RECOGSIZE);
        filePath = datahandler.GetFilePath(wordId, 0, 1, "wav");
        filePath = path.append(filePath);

        recognizer.clear();
        do
        {
            realSize = datahandler.ReadWav(filePath, littleVoiceBuffer, 2000, position);
            bcontinue = mfcc.AddBuffer(littleVoiceBuffer, realSize);
            position += realSize;
            if(realSize != 2000) bcontinue = false;
            if(position > 8000)
            {
                name = gmm.Classify(mfcc.GetMFCCData(), mfcc.GetFrameCount());
                recognizer[name]++;
            }
        } while(bcontinue);

        auto it1 = max_element(recognizer.cbegin(), recognizer.cend(), [](const std::pair<std::string, int>& p1, const std::pair<std::string, int>& p2) { return p1.second < p2.second; });
        // Get name from model
        name = it1->first;
        // Get score from model
        int score = it1->second;
        recognizer[name] = 0; //To find the second
        it1 = max_element(recognizer.cbegin(), recognizer.cend(), [](const std::pair<std::string, int>& p1, const std::pair<std::string, int>& p2) { return p1.second < p2.second; });

        std::cout << " " << name << " " << score*100/(score+it1->second) << "%";

        if(name == datahandler.GetWord(wordId))
        {
            //name = GetWord(name);
            std::cout << " correctly : " << GetWord(name) << std::endl;
        }
        else
        {
            //name = GetWord(name);
            if(GetWord(name) == " ")
            {
                omissions++;
            }
            else
            {
                replacements++;
            }
            std::cout << " wrong : " << GetWord(name) << std::endl;
            wrong_word++;
        }

        filePath.erase();
    }
    recogPercentEnd = Clock::now();
    std::cout << std::endl;

    ms = std::chrono::duration_cast<Milliseconds>(trainEnd - trainStart);
   	std::cout << "Training time " << ms.count() << " ms (" << ms.count()/8 << " by train)" << std::endl;
    ms = std::chrono::duration_cast<Milliseconds>(recogScoreEnd - recogScoreStart);
	std::cout << "Recognition by score time " << ms.count() << " ms (" << ms.count()/24 << " by recog)" << std::endl;
    ms = std::chrono::duration_cast<Milliseconds>(recogPercentEnd - recogPercentStart);
	std::cout << "Recognition by % time " << ms.count() << " ms (" << ms.count()/24 << " by recog)" << std::endl;

    // Calculate the Word error rate
    double wer = (replacements + omissions + insertions)* 100 / NUM_WORDS;
    std::cout << std::endl << std::endl;
    std::cout << "Word Error Rate (WER): " << wer << "%" << std::endl;

    // Calculate the word accuracy
    double wa = (NUM_WORDS - wrong_word) * 100/ NUM_WORDS;
    std::cout << "Word Accuracy (WA): " << wa << "%" << std::endl;
        std::cout << std::endl << std::endl;
    return 0;
}

std::string GetWord(std::string name)
{
    if(name == "Word0") return "Null";
    if(name == "Word1") return "Eins";
    if(name == "Word2") return "Zwei";
    if(name == "Word3") return "Drei";
    if(name == "Word4") return "Vier";
    if(name == "Word5") return "Fünf";
    if(name == "Word6") return "Sechs";
    if(name == "Word7") return "Sieben";
    if(name == "Word8") return "Acht";
    if(name == "Word9") return "Neun";
    if(name == "Word10") return "Silence";
    if(name == "Word11") return "An";
    if(name == "Word12") return "Aus";
    if(name == "Word13") return "Stopp";
    if(name == "Word14") return "Hallo";
    if(name == "Word15") return "Licht";
    if(name == "Word16") return "Weiter";
    else return " ";
    return " ";
}
