#pragma once
#include <vector>
#include "opencv/cv.h"
#include "SoundUtils.h"
#include "Microphone.h"

class Training
{
public:
	Training(void);
	~Training(void);
	int Train(int numberOfTrainingSamples, char * pData, int dataSize);
	int LoadFromTrainingData(int numberOfTrainingSamples, double * pData, int dataSize);
	int Classify(char * pData, int dataSize);

	std::vector<CvMat *> X;
	std::vector<CvMat *> C;
	SoundUtils sound;
	CvMat * XM;
	CvMat * CM;
	CvMat * w0;
	CvMat * wa;
	CvMat * w1;
	CvMat * wb;
	CvMat * ukX;
	CvMat * ukC;
};
