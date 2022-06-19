#pragma once
#include "stdafx.h"
class SoundUtils
{
public:
	SoundUtils(void);
	~SoundUtils(void);
	CvMat* ExtractImpactSound(CvMat * pData, int nSamplingRate);
	CvMat* CalcHannWindow(int nWindowSize);
	double Norm(CvMat *Data, bool normMatrix = true);
	CvMat* MelCeptrum(CvMat* magnitudes, CvMat *freqs );
	CvMat* ExtractBands(CvMat* magnitudes, CvMat *freqs,
		double startFreq, double breakPoint, double endFreq, int numberOfPieces);
	CvMat* FindIndices(CvMat* x, CvMat* values );
	CvMat* Energy(CvMat * x);
	CvMat* Cepstrum(CvMat * e, int k);

	/// Doðru kayýt yapýlan ses sayýsýný döner
	int findFeatureVectors();
	int pc_evectors(CvMat *A,int numvecs, /*CvMat* Vectors, CvMat *Values,*/CvMat *Psi, int type);
	void sortem(CvMat *Vectors,CvMat *Values);

	CvMat *stdVector2CvMat(std::vector<CvMat*> vectors);

	CvMat * ColumnMeans(CvMat* AMatrix);

	CvMat * Transpose(CvMat* AMatrix);

	CvMat * Diagonal(CvMat* AMatrix);

	CvMat * GetColVector(CvMat* AMatrix, int nCol);
	CvMat * GetColVectors(CvMat* AMatrix, int start, int end);

	CvMat * MatrixMul(CvMat* AMatrix, CvMat* BMatrix);

	static void PrintMatrix(CvMat * AMatrix);

	CvMat * FFT(CvMat * Input, int nSize);

	int Train(int numberOfTrainingSamples, short * pData, int dataSize);
	int LoadFromImpactData(int numberOfTrainingSamples, double * pData, int dataSize);
	int Classify(short * pData, int dataSize);

	std::vector<CvMat *> X;
	std::vector<CvMat *> C;
	std::vector<CvMat *>X_Data;

	CvMat * XM;
	CvMat * XM_Data;
	CvMat * CM;

	CvMat * w0;
	CvMat * wa;
	CvMat * w1;
	CvMat * wb;
	CvMat * ukX;
	CvMat * ukC;

	/// PSI Vectors
	CvMat *Xm; 
	CvMat *Cm;

	CvMat * ValuesX;
	CvMat * ValuesC;

	int openFistik;
	int closedFistik;

	int nImpactSizeCounter;

	CvMat* impactSound;
};
