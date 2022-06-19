#include "stdafx.h"
#include "Training.h"
#include "SoundUtils.h"


#include <al.h>
#include <alc.h>

Training::Training(void)
{
}

Training::~Training(void)
{
}

int Training::Train(int numberOfTrainingSamples, char * pData, int dataSize)
{
	
	if (X.size()<numberOfTrainingSamples)
	{

		//int trainedSamples = X.size();

		///*mic.Read(pData, samplingRate*5);*/

		//
		//double pData1[192000*5];
		//for (int i = 0; i < 192000*5; i++ )
		//{
		//	pData1[i] = pData[i]/128.0;
		//}


		CvMat *soundMatrix = cvCreateMat(samplingRate*5, 1, CV_64FC1);
		*soundMatrix=cvMat(samplingRate*5, 1, CV_64FC1, pData);
		
		CvMat *pImpactSound;
		pImpactSound = cvCreateMat(1,288, CV_64FC1);
		*soundMatrix=cvMat(1,288, CV_64FC1, pData);
		//pImpactSound = sound.ExtractImpactSound(soundMatrix, samplingRate);
		//for (int i = 0; i < 288; i++)
		//{
		//	cvmSet(pImpactSound, 0, i, 14+trainedSamples+i);
		//}
		sound.PrintMatrix(pImpactSound);

		CvSize pImpactSize = cvGetSize(pImpactSound);
		if(pImpactSize.width>0)
		{

			CvMat *hannWindow = sound.CalcHannWindow( pImpactSound->cols);
			
			CvMat* pImpactSoundTr = sound.Transpose(pImpactSound);
			
			cvMul(pImpactSoundTr, hannWindow, pImpactSoundTr);
			CvMat * pOutput = sound.FFT(pImpactSoundTr, 256);
			
			CvMat* freq = 
				cvCreateMat(pOutput->rows, pOutput->cols, CV_64FC1);
			cvAbs(pOutput, freq );

			CvMat* freqs = cvCreateMat(1,129,CV_64FC1);
			CvMat* magnitudes = cvCreateMat(129,1,CV_64FC1);
			for(int i=0; i<129; i++)
			{
				double fFreq = ((double)samplingRate)*i/256;
				cvmSet(freqs, 0, i, fFreq);
				double fMag = cvmGet(freq, i, 0);
				cvmSet(magnitudes, i, 0, fMag );
			}

			sound.Norm(magnitudes);
			sound.Norm(pImpactSound);

			X.push_back(sound.Transpose(pImpactSound));
			CvMat *impactCepstrum = sound.MelCeptrum(magnitudes, freqs);
			sound.PrintMatrix(impactCepstrum);
			sound.Norm(impactCepstrum);
			sound.PrintMatrix(impactCepstrum);
			C.push_back(sound.Transpose(impactCepstrum));
		}

		// 10 düzgün ses olduðunda
		if (X.size() >= 40)
		{
			XM = sound.stdVector2CvMat(X);
			CM = sound.stdVector2CvMat(C);

			sound.PrintMatrix(CM);
			int numGoodInputs = sound.findFeatureVectors();

			if (numGoodInputs < 40)
			{

				// Sorunlu sesleri pop back et
				for (int i = 0; i < numGoodInputs; i++)
				{
					X.pop_back();
					C.pop_back();
				}
			} 
		}
	}
	

	return X.size();
}

int Training::Classify( char * pData, int dataSize )
{
	return 0;
}

int Training::LoadFromTrainingData(int numberOfTrainingSamples, double * pData, int dataSize)
{

	if (X.size()<numberOfTrainingSamples)
	{

		//int trainedSamples = X.size();

		///*mic.Read(pData, samplingRate*5);*/

		//
		//double pData1[192000*5];
		//for (int i = 0; i < 192000*5; i++ )
		//{
		//	pData1[i] = pData[i]/128.0;
		//}


		CvMat *soundMatrix = cvCreateMat(samplingRate*5, 1, CV_64FC1);
		*soundMatrix=cvMat(samplingRate*5, 1, CV_64FC1, pData);

		CvMat *pImpactSound;
		pImpactSound = cvCreateMat(1,288, CV_64FC1);
		*pImpactSound=cvMat(1,288, CV_64FC1, pData);
		//pImpactSound = sound.ExtractImpactSound(soundMatrix, samplingRate);
		//for (int i = 0; i < 288; i++)
		//{
		//	cvmSet(pImpactSound, 0, i, 14+trainedSamples+i);
		//}
		sound.PrintMatrix(pImpactSound);

		CvSize pImpactSize = cvGetSize(pImpactSound);
		if(pImpactSize.width>0)
		{

			CvMat *hannWindow = sound.CalcHannWindow( pImpactSound->cols);

			CvMat* pImpactSoundTr = sound.Transpose(pImpactSound);

			cvMul(pImpactSoundTr, hannWindow, pImpactSoundTr);
			CvMat * pOutput = sound.FFT(pImpactSoundTr, 256);

			CvMat* freq = 
				cvCreateMat(pOutput->rows, pOutput->cols, CV_64FC1);
			cvAbs(pOutput, freq );

			CvMat* freqs = cvCreateMat(1,129,CV_64FC1);
			CvMat* magnitudes = cvCreateMat(129,1,CV_64FC1);
			for(int i=0; i<129; i++)
			{
				double fFreq = ((double)samplingRate)*i/256;
				cvmSet(freqs, 0, i, fFreq);
				double fMag = cvmGet(freq, i, 0);
				cvmSet(magnitudes, i, 0, fMag );
			}

			sound.Norm(magnitudes);
			sound.Norm(pImpactSound);

			X.push_back(sound.Transpose(pImpactSound));
			CvMat *impactCepstrum = sound.MelCeptrum(magnitudes, freqs);
			sound.PrintMatrix(impactCepstrum);
			sound.Norm(impactCepstrum);
			sound.PrintMatrix(impactCepstrum);
			C.push_back(sound.Transpose(impactCepstrum));
		}

		// 10 düzgün ses olduðunda
		if (X.size() >= 40)
		{
			XM = sound.stdVector2CvMat(X);
			CM = sound.stdVector2CvMat(C);

			CvMat *Xm = 0; CvMat *Cm = 0;

			sound.PrintMatrix(CM);
			int numGoodInputs = sound.findFeatureVectors();

			sound.PrintMatrix(w0);
			sound.PrintMatrix(wa);
			sound.PrintMatrix(w1);
			sound.PrintMatrix(wb);
			sound.PrintMatrix(ukX);
			sound.PrintMatrix(ukC);


			if (numGoodInputs < 40)
			{

				// Sorunlu sesleri pop back et
				for (int i = 0; i < numGoodInputs; i++)
				{
					X.pop_back();
					C.pop_back();
				}
			} 
		}
	}


	return X.size();
}
