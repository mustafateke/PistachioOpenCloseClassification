#include "stdafx.h"
#include "SoundUtils.h"

SoundUtils::SoundUtils(void)
{
	openFistik = 0;
	closedFistik = 0;
	nImpactSizeCounter = 0;
}

SoundUtils::~SoundUtils(void)
{
}

CvMat* SoundUtils::ExtractImpactSound( CvMat * pData, int nSamplingRate )
{
	//Extract impact sound
	int startIndex = -1;
	CvSize size = cvGetSize(pData);
	for (int i=0; i <size.width; i++)
	{
		float data = cvmGet(pData,0,i);
		if(data > 0.085)
		{
			startIndex = i;
			break;
		}
	}

	int nMax = cvCeil(nSamplingRate * 0.0015);
	if (nImpactSizeCounter%nMax == 0)
	{
		if(impactSound) cvReleaseMat(&impactSound);
		impactSound = cvCreateMat(1,nMax,CV_64FC1);
		nImpactSizeCounter = 0;
	}	
	else
	{
		startIndex = 0;
	}

	if (startIndex + nMax > size.width)
	{
		nMax = size.width - startIndex;
	}
	if (startIndex >= 0)
	{
		for (nImpactSizeCounter; nImpactSizeCounter< nMax ; nImpactSizeCounter++)
		{
			double fVal = cvmGet(pData,0,startIndex + nImpactSizeCounter);
			cvmSet(impactSound,0, nImpactSizeCounter, fVal);
		}

		if (nImpactSizeCounter%nMax == 0)
		{
			return impactSound;
		}	
		else
		{
			return 0;
		}

	}
	else return 0;
}

CvMat* SoundUtils::CalcHannWindow( int nWindowSize )
{
	float a0 = 0.5;
	float a1 = 0.5;
	float a2 = 0;
	float a3 = 0;
	float a4 = 0;

	int half = 0;
	if (nWindowSize%2==0){// Even length window
		half = nWindowSize/2;
	}
	else{// Odd length window
		half = (nWindowSize+1)/2;
	}

	
	CvMat* w = cvCreateMat(nWindowSize,1,CV_64FC1);

	for(int i = 0; i < half; i++)
	{
		double x = i/(double(nWindowSize-1));
		double fw = a0 - a1*cos(2*PI*x);
		cvmSet(w,i, 0, fw);
		cvmSet(w,nWindowSize -i-1, 0, fw);
	}
	return w;

}

double SoundUtils::Norm( CvMat *Data, bool normMatrix )
{
	CvSize size = cvGetSize(Data);
	double fNorm = 0;
	double fSquareSum = 0;
	for (int y = 0; y < size.height; y++)
		for(int x = 0; x < size.width; x++ )
		{
			double fVal = cvmGet(Data, y, x);
			fSquareSum += fVal*fVal;
		}

		fNorm = std::sqrt(fSquareSum);
		if (normMatrix)
		{		
			for (int y = 0; y < size.height; y++)
				for(int x = 0; x < size.width; x++ )
				{
					double fVal = cvmGet(Data, y, x);
					fVal = abs(fVal)/fNorm;
					cvmSet(Data,y, x, fVal);
				}
		}
	return fNorm;
}

CvMat* SoundUtils::MelCeptrum( CvMat* magnitudes, CvMat* freqs )
{
	//PrintMatrix(freqs);
	CvMat *bands = ExtractBands(magnitudes, freqs, 0, 20000, 44000, 24 );
	//PrintMatrix(bands);
	CvMat *energies = Energy(bands);
	//PrintMatrix(energies);
	CvMat * c = Cepstrum(energies, 20);
	cvReleaseMat(&energies);
	cvReleaseMat(&bands);
	return c;
}

CvMat* SoundUtils::ExtractBands( CvMat* magnitudes, CvMat *freqs,
								double startFreq, double breakPoint, double endFreq, int numberOfPieces )
{
	int sizeValues;
	sizeValues= 2*((numberOfPieces)/2) +1;
	CvMat * valuesLinear = cvCreateMat(1, 1+numberOfPieces/2, CV_64FC1);
	CvMat * valuesLogarithmic = cvCreateMat(1, 1+numberOfPieces/2, CV_64FC1);
	CvMat * boundaries = cvCreateMat(1,sizeValues , CV_64FC1);


	for (int i = 0; i < 1+ numberOfPieces/2; i++)
	{
		double widthOfAPiece = (breakPoint - startFreq)/(numberOfPieces/2);
		double fLinearVal = (i * widthOfAPiece) + startFreq;
		cvmSet(valuesLinear, 0, i, fLinearVal);
		cvmSet(boundaries, 0, i, fLinearVal);
	}

	for (int i = 0; i <= numberOfPieces/2; i++)
	{
		double widthOfAPiece = (log10(endFreq) -log10(breakPoint))/(numberOfPieces/2);
		double valuesInLogScale = (i * widthOfAPiece) + log10(breakPoint);
		double fLogVal = pow(10, valuesInLogScale);
		cvmSet(valuesLogarithmic, 0, i, fLogVal);
		cvmSet(boundaries, 0, (numberOfPieces/2)+i, fLogVal);
	}


	CvMat *boundaryIndices = FindIndices(freqs, boundaries);

	//PrintMatrix(boundaryIndices);
	CvSize szBoundaryIndices = cvGetSize(boundaryIndices);
	CvSize szMagnitudes = cvGetSize(magnitudes);
	CvMat *bands = cvCreateMat(szBoundaryIndices.width-1, szMagnitudes.height, CV_64FC1);
	cvSet(bands, cvScalar(0)); // Kontrol Et : Sorunsuz
	//PrintMatrix(magnitudes);
	for (int i = 0; i < szBoundaryIndices.width-1; i++)
	{
		int l = 0;
		int start = 0 ;
		if (i == 0)
		{
			l = cvmGet(boundaryIndices, 0, i+1) - cvmGet(boundaryIndices, 0, i) + 1;
			start = cvmGet(boundaryIndices, 0, i) ;
		} 
		else
		{
			l = cvmGet(boundaryIndices, 0, i+1) - cvmGet(boundaryIndices, 0, i);
			start = cvmGet(boundaryIndices, 0, i)  + 1;
		}
		for (int j = 0; j < l; j++)
		{
			double fMag = cvmGet(magnitudes,start+ j,0);
			cvmSet(bands, i, j, fMag );

		}
	}
	//PrintMatrix(bands);
	cvReleaseMat(&valuesLinear);
	cvReleaseMat(&valuesLogarithmic);
	cvReleaseMat(&boundaries);
	cvReleaseMat(&boundaryIndices);
	return bands;
}

CvMat* SoundUtils::FindIndices( CvMat* x, CvMat* values )
{

	CvSize xSize = cvGetSize(x);
	CvSize valuesSize = cvGetSize(values);

	CvMat *indices = cvCreateMat(1, valuesSize.width, CV_64FC1);

	for (int i = 0; i < valuesSize.width; i++)
	{
		double fValue = cvmGet(values, 0, i);
		for (int j = 0; j < xSize.width; j++)
		{
			double fX = cvmGet(x, 0, j);
			if (fX+0.00001 > fValue)// Deðerlerin ayný olmasý durumunda eþitliði bozmak için
			{
				cvmSet(indices, 0, i, j);
				break;
			}
		}
	}

	return indices;
}

CvMat* SoundUtils::Energy( CvMat * x )
{
	CvSize size =cvGetSize(x);
	CvMat *xt = cvCreateMat(size.width, size.height, CV_64FC1);
	CvMat *e = cvCreateMat(1, size.height, CV_64FC1);
	cvTranspose(x, xt);

	CvSize sizet = cvGetSize(xt);
	for (int x = 0; x < sizet.width; x++)
	{
		double rowSum = 0;
		for(int y = 0; y < sizet.height; y++ )
		{
			double fVal = cvmGet(xt, y, x);
			rowSum += fVal*fVal;

		}
		cvmSet(e,0, x, rowSum);
	}

	cvReleaseMat(&xt);

	return e;
}

CvMat* SoundUtils::Cepstrum( CvMat * e, int k )
{
	CvSize size = cvGetSize(e);
	CvMat *c = cvCreateMat(1, k, CV_64FC1);
	for (int j = 0;j < k; j++)
	{
		double fSum = 0;
		for (int i = 0; i < size.width; i++)
		{
			double fe = cvmGet(e, 0, i);
			fSum +=  ( log10(fe) * cos((j+1) * ((i+1) - 0.5) * PI / size.width) ) ;
		}
		cvmSet(c,0, j, fSum );
	}
	//PrintMatrix(c);
	return c;
}

int SoundUtils::findFeatureVectors()
{
	ukX = NULL;
	ukC = NULL;

	int xNumber  = pc_evectors( XM,10,/*ukX, ValuesX,*/Xm, 0 );
	int cNumber =pc_evectors( CM,10,/*ukC, ValuesC,*/Cm, 1 );

	//PrintMatrix(ukX);
	//PrintMatrix(ukC);
	int minNumber = (xNumber>cNumber?cNumber:xNumber);
	if (minNumber == 10)
	{

		w0 = cvCreateMat(10, 1, CV_64FC1); cvSet(w0, cvScalar(0));
		wa = cvCreateMat(10, 1, CV_64FC1); cvSet(w0, cvScalar(0));
		w1 = cvCreateMat(10, 1, CV_64FC1); cvSet(w0, cvScalar(0));
		wb = cvCreateMat(10, 1, CV_64FC1); cvSet(w0, cvScalar(0));

		for (int j= 0; j < 20; j++ )
		{
			cvAdd( w0, MatrixMul(Transpose(ukX), GetColVector(XM, j)), w0);
			cvAdd( wa, MatrixMul(Transpose(ukC), GetColVector(CM, j)), wa);
		}

		//PrintMatrix(w0);
		cvCvtScale(w0, w0, 1/20.0);
		//PrintMatrix(w0);
		cvCvtScale(wa, wa, 1/20.0);
		//PrintMatrix(wa);

		for (int j= 20; j < 40; j++ )
		{
			cvAdd( w1, MatrixMul(Transpose(ukX), GetColVector(XM, j)), w1);
			cvAdd( wb, MatrixMul(Transpose(ukC), GetColVector(CM, j)), wb);
		}

		//PrintMatrix(w1);
		//PrintMatrix(wb);
		cvCvtScale(w1, w1, 1/20.0);
		cvCvtScale(wb, wb, 1/20.0);
	}
	return minNumber;
}

int SoundUtils::pc_evectors( CvMat *A,int numvecs,CvMat *Psi, int type )
{
	CvSize sizeA = cvGetSize(A);
	//PrintMatrix(A);
	int nexamp = sizeA.width;
	CvMat *Vectors;
	CvMat *Values;
	if(type == 0)
	{
		Vectors = cvCreateMat(A->cols, A->cols, CV_64FC1);
		Values = cvCreateMat(XM->cols, 1, CV_64FC1);
	}
	else {
		Vectors = cvCreateMat(A->cols, CM->cols, CV_64FC1);
		Values = cvCreateMat(A->cols, 1, CV_64FC1);
	}
	Psi= Transpose(ColumnMeans(Transpose(A))); // Calculate mean for each row


	//PrintMatrix(Psi);
	for (int i=0; i < nexamp; i++)
	{
		for (int j = 0; j < sizeA.height; j++)
		{
			double fVal = cvmGet(A,j,i ) -cvmGet(Psi,j, 0) ;
			cvmSet(A, j, i, fVal);
		}
	}

	//PrintMatrix(A);

	/*PrintMatrix(A);*/
	CvMat *At = Transpose(A);
	/*PrintMatrix(At);*/
	CvMat *L = cvCreateMat(sizeA.width, sizeA.width, CV_64FC1);
	//PrintMatrix(L);
	
	//PrintMatrix(At);
	cvMatMul(At, A, L);
	/*PrintMatrix(L);*/
	//PrintMatrix(L);
	
	//CvMat * singularvalues = cvCreateMat(sizeA.width, sizeA.width, CV_64FC1);
	//CvMat * U = cvCreateMat(sizeA.width, sizeA.width, CV_64FC1);
	//CvMat * V = cvCreateMat(sizeA.width, sizeA.width, CV_64FC1);

	//cvSVD(L, singularvalues, U, V);

	//PrintMatrix(singularvalues);
	//PrintMatrix(U);
	//PrintMatrix(V);
	

	cvEigenVV(L, Vectors, Values );
	PrintMatrix(Vectors);
	Vectors = Transpose(Vectors);
	//cvAbs(Transpose(Vectors), Vectors);

	//sortem(Vectors, Values); // no need anymore since opencv already sorts
	//PrintMatrix(Vectors);
	
	CvMat * Vectors2 = cvCreateMat(A->height, Vectors->height, CV_64FC1);
	//PrintMatrix(A);
	cvMatMul(A, Vectors, Vectors2);
	
	Vectors = Vectors2;
	//PrintMatrix(Values);
	//PrintMatrix(Vectors);
	//Values = Diagonal(Values);
	//PrintMatrix(Values);
	cvCvtScale(Values, Values, 1.0/(nexamp-1));

	int num_good = 0; // good sample counter

	for (int i = 0; i < nexamp; i++)
	{
		CvSize sizeVec = cvGetSize(Vectors);
		double normVec = Norm(GetColVector(Vectors, i), false); // check this
		double fValValue = cvmGet(Values, i, 0);
		double ratio = 1;
		if (fValValue < 0.00001)
		{
			cvmSet(Values, i, 0, 0);
			ratio = 0;
		}
		else
		{
			num_good++;
		}
		for(int row = 0; row < sizeVec.height; row++)
		{
			double fVal = cvmGet(Vectors, row, i);
			fVal = fVal/normVec;
			cvmSet(Vectors, row, i, ratio*fVal);
		}
	}
	if (numvecs >num_good )
	{
		printf("Warning: numvecs is %d; only %d exist.\n",numvecs,num_good);
		numvecs = num_good;
	}
	//PrintMatrix(Vectors);
	if (numvecs > 0)
	{
		Vectors = GetColVectors(Vectors, 0, numvecs);
	}

	if(type == 0)
	{
		ukX = Vectors;
		//PrintMatrix(Vectors);
		ValuesX = Values;
		Xm = Psi;
	}
	else {
		ukC = Vectors;
		ValuesC = Values;
		Cm = Psi;
	}

	return numvecs;

	/// Numgood stuff
}

void SoundUtils::sortem( CvMat *Vectors,CvMat *Values )
{
	CvMat *D = Diagonal(Values); //Check
	CvSize Vecsize = cvGetSize(Vectors);
	CvSize Valsize = cvGetSize(Vectors);

	CvMat *NV = cvCreateMat(Vecsize.height, Vecsize.width, CV_64FC1);
	CvMat *ND = cvCreateMat(Valsize.height, Valsize.width, CV_64FC1);

	CvMat *dvec = cvCreateMat(Vecsize.height, Vecsize.width, CV_64FC1);
	CvMat *index_dv = cvCreateMat(Vecsize.height, Vecsize.width, CV_64FC1);

	cvSet(NV, cvScalar(0.0)); // Kontrol Et
	cvSet(ND, cvScalar(0.0)); // Kontrol Et

	cvSort(D, dvec, index_dv); // Check for FlipUD; Ascending veya descending olmasý kontrol edilebilir.
	for (int col = 0; col< Valsize.height; col++)
	{
		int nvindex = cvmGet(index_dv, col, 0);
		double fDVal = cvmGet(Values, nvindex, nvindex) ;
		cvmSet(ND, nvindex, nvindex, fDVal);
		for (int row = 0; row < Valsize.width; row++)
		{
			double fVVal = cvmGet(Vectors, row, nvindex);
			cvmSet(NV,row,nvindex, fVVal );
		}
	}

	Values = ND;
	Vectors = NV;

}

CvMat * SoundUtils::stdVector2CvMat( std::vector<CvMat*> vectors )
{
	int numcols = vectors.size();
	CvSize size;
	if (numcols > 0)
	{
		size = cvGetSize(vectors[0]);
	}
	CvMat * AMatrix = cvCreateMat(size.height, numcols, CV_64FC1);
	for(int col = 0; col < numcols; col++)
	{
		for (int row = 0; row < size.height; row++)
		{
			double fVal = cvmGet(vectors[col],row, 0);
			cvmSet(AMatrix, row, col, fVal);
		}
	}

	return AMatrix;
}

CvMat * SoundUtils::ColumnMeans( CvMat* AMatrix )
{
	CvSize size = cvGetSize(AMatrix);
	CvMat *Means = cvCreateMat(1, size.width, CV_64FC1);
	for (int col = 0; col< size.width; col++)
	{
		double colSum = 0;
		for(int row = 0; row< size.height; row++)
		{
			double fVal = cvmGet(AMatrix, row, col);
			colSum += fVal;
		}
		double colMean = colSum/size.height;
		cvmSet(Means, 0, col, colMean);
	}
	return Means;
}

CvMat * SoundUtils::Transpose( CvMat* AMatrix )
{
	CvSize size = cvGetSize(AMatrix);
	CvMat *At = cvCreateMat(size.width, size.height, CV_64FC1);
	cvTranspose(AMatrix, At);
	return At;
}

CvMat * SoundUtils::Diagonal( CvMat* AMatrix )
{
	CvSize size = cvGetSize(AMatrix);
	int numDiags = size.height;

	CvMat *Diags = cvCreateMat(numDiags, 1, CV_64FC1);

	for (int i = 0; i < numDiags; i++)
	{
		double fVal = cvmGet(AMatrix, i, i);
		cvmSet(Diags, i, 0, fVal);
	}
	return Diags;
}

CvMat * SoundUtils::GetColVector( CvMat* AMatrix, int nCol )
{
	CvSize size = cvGetSize(AMatrix);
	CvMat * ColVector = cvCreateMat(size.height, 1, CV_64FC1);
	cvGetCol(AMatrix, ColVector, nCol);
	return ColVector;
}

CvMat * SoundUtils::GetColVectors( CvMat* AMatrix, int start, int end )
{
	CvSize size = cvGetSize(AMatrix);
	CvMat * ColVector = cvCreateMat(size.height, end - start, CV_64FC1);
	cvGetCols(AMatrix, ColVector, start, end);
	return ColVector;
}

CvMat * SoundUtils::MatrixMul( CvMat* AMatrix, CvMat* BMatrix )
{
	CvSize sizeA = cvGetSize(AMatrix);
	CvSize sizeB = cvGetSize(BMatrix);


	CvMat *Result = cvCreateMat(sizeA.height, sizeB.width, CV_64FC1);
	cvMatMul(AMatrix, BMatrix, Result);

	return Result;
}

/// Matrixleri görmek için debug kodu
void SoundUtils::PrintMatrix( CvMat * AMatrix )
{
	FILE * pFile;

	char name [100];

	pFile = fopen ("mat.txt","w");


	CvSize size = cvGetSize(AMatrix);
	fprintf(pFile,"**************\n");
	for (int row = 0; row<size.height;row++)
	{
		for (int col = 0; col < size.width;col++)
		{
			fprintf(pFile, "%2.8f ", cvmGet(AMatrix, row, col));
		}
		fprintf(pFile,"\n");
	}

	fclose (pFile);
}

/// OpenCV'nin FFT'si ile Matlab'ýnki farklý
/// ikisini eþitlemek için bu dönüþümler gerekli
CvMat * SoundUtils::FFT( CvMat * Input, int nSize )
{
	CvMat* pInput = cvCreateMat((nSize),Input->cols, CV_64FC1);
	for (int row = 0; row < nSize; row++)
	{
		cvmSet(pInput, row, 0, cvmGet(Input, row, 0));
	}
	CvMat *pOutput = cvCreateMat((nSize),Input->cols, CV_64FC1);

	cvDFT(pInput, pOutput, CV_DXT_FORWARD);

	CvMat *pOutputABS = cvCreateMat((nSize),Input->cols, CV_64FC1);

	double DC = cvmGet(pOutput, 0, 0);
	cvmSet(pOutputABS, 0, 0, abs(DC));

	double mid = cvmGet(pOutput, 255, 0);
	cvmSet(pOutputABS, 128, 0, abs(mid));

	for (int row = 1; row < 128; row++)
	{
		double realVal = cvmGet(pOutput, 2*(row-1)+1, 0);
		double imgVal = cvmGet(pOutput, 2*(row-1)+2, 0);
		double fVal = sqrt(realVal*realVal + imgVal*imgVal);
		cvmSet(pOutputABS, row, 0, fVal);
		cvmSet(pOutputABS, 256-row, 0, fVal);
	}
	cvReleaseMat(&pInput);
	cvReleaseMat(&pOutput);

	return pOutputABS;
}

int SoundUtils::Train(int numberOfTrainingSamples, short * pData, int dataSize)
{

	if (X.size()<numberOfTrainingSamples)
	{

		CvMat *soundMatrix = cvCreateMat(samplingRate*5, 1, CV_64FC1);
		*soundMatrix=cvMat(samplingRate*recordDuration, 1, CV_64FC1, pData);

		/// Windows Ses deðerlerini -127 ile 128 arasýnda okuyor
		cvCvtScale(soundMatrix, soundMatrix, 1/sesMax);

		CvMat *pImpactSound;
		/*pImpactSound = cvCreateMat(1,288, CV_64FC1);*/
		*soundMatrix=cvMat(1,288, CV_64FC1, pData);

		pImpactSound = ExtractImpactSound(soundMatrix, samplingRate);
		
		cvReleaseMat(&soundMatrix);

		if(pImpactSound != 0)
		{

			CvSize pImpactSize = cvGetSize(pImpactSound);
			//PrintMatrix(pImpactSound);
			CvMat *hannWindow = CalcHannWindow( pImpactSound->cols);

			CvMat* pImpactSoundTr = Transpose(pImpactSound);
			X_Data.push_back(Transpose(pImpactSound));

			cvMul(pImpactSoundTr, hannWindow, pImpactSoundTr);
			CvMat * pOutput = FFT(pImpactSoundTr, 256);

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

			Norm(magnitudes);
			Norm(pImpactSound);

			X.push_back(Transpose(pImpactSound));
			X_Data.push_back(Transpose(pImpactSound));

			CvMat *impactCepstrum = MelCeptrum(magnitudes, freqs);
			//PrintMatrix(impactCepstrum);
			Norm(impactCepstrum);
			//PrintMatrix(impactCepstrum);
			C.push_back(Transpose(impactCepstrum));

		}

		// 10 düzgün ses olduðunda
		if (X.size() >= 40)
		{
			XM = stdVector2CvMat(X);
			CM = stdVector2CvMat(C);
			XM_Data =stdVector2CvMat(X_Data);

			CvMat *Xm = 0; CvMat *Cm = 0;

			//PrintMatrix(CM);
			int numGoodInputs = findFeatureVectors();

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


int SoundUtils::Classify( short * pData, int dataSize )
{
	CvMat *soundMatrix = cvCreateMat( 1,samplingRate/nFistikPerSec,  CV_64FC1);
	//*soundMatrix=cvMat(samplingRate/nFistikPerSec, 1, CV_64FC1, pData);
	//PrintMatrix(soundMatrix);
	/// Windows Ses deðerlerini -127 ile 128 arasýnda okuyor
	//double* pDataD = new double[dataSize];
	for (int i = 0; i < dataSize; i++)
	{
		double fVal = pData[i]/sesMax;
		cvmSet(soundMatrix, 0, i ,fVal );
	}
	//cvCvtScale(soundMatrix, soundMatrix, 1/128.0);

	CvMat *pImpactSound;
	//pImpactSound = cvCreateMat(1,dataSize, CV_64FC1);
	//*soundMatrix=cvMat(1,dataSize, CV_64FC1, pDataD);

	
	pImpactSound = ExtractImpactSound(soundMatrix, samplingRate);
	
	

	if(pImpactSound != 0)
	{
		PrintMatrix(soundMatrix);

		CvSize pImpactSize = cvGetSize(pImpactSound);
		PrintMatrix(pImpactSound);
		CvMat *hannWindow = CalcHannWindow( pImpactSound->cols);

		CvMat* pImpactSoundTr = Transpose(pImpactSound);

		cvMul(pImpactSoundTr, hannWindow, pImpactSoundTr);
		CvMat * pOutput = FFT(pImpactSoundTr, 256);
		//PrintMatrix(pOutput);
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

		//PrintMatrix(pImpactSound);
		Norm(magnitudes);
		//PrintMatrix(magnitudes);
		Norm(pImpactSound);

		CvMat *impactCepstrum = MelCeptrum(magnitudes, freqs);
		//PrintMatrix(impactCepstrum);
		Norm(impactCepstrum);
		//PrintMatrix(impactCepstrum);

		CvMat * impSoundTr = Transpose(pImpactSound);
		//PrintMatrix(pImpactSound);
		cvSub(impSoundTr, Xm, impSoundTr);
		//PrintMatrix(Xm);
		//PrintMatrix(impSoundTr);
		CvMat * ukXTr = Transpose(ukX);
		CvMat * wx = MatrixMul( ukXTr,  impSoundTr);
		cvReleaseMat(&ukXTr);

		//PrintMatrix(Xm);
		//PrintMatrix(ukX);
		//PrintMatrix(wx);
		//PrintMatrix(ukC);

		//PrintMatrix(impactCepstrum);
		CvMat * impCepstrumTr = Transpose(impactCepstrum);
		//PrintMatrix(impCepstrumTr);
		cvSub(impCepstrumTr, Cm, impCepstrumTr);

		CvMat * ukCTr = Transpose(ukC);
		//PrintMatrix(ukCTr);

		CvMat * wc = MatrixMul( ukCTr,  impCepstrumTr);

		//PrintMatrix(wc);
		/// open için
		CvMat *wx0 = cvCreateMat(wx->rows, wx->cols, CV_64FC1);
		cvSub(wx, w0, wx0); 
		double fwx0 = Norm(wx0);
		//PrintMatrix(wx0);
		CvMat *wca = cvCreateMat(wc->rows, wc->cols, CV_64FC1);
		//PrintMatrix(wa);
		//PrintMatrix(wc);
		cvSub(wc, wa, wca); 
		//PrintMatrix(wca);
		double fwca = Norm(wca);
		//PrintMatrix(wca);
		//cvAdd(wx0, wca, wx0);
		double distanceOpen = fwx0 + fwca;//cvmGet(wx0, 0, 0);

		/// closed için
		CvMat *wx1 = cvCreateMat(wx->rows, wx->cols, CV_64FC1);
		cvSub(wx, w1, wx1); 
		double fwx1 = Norm(wx1);

		CvMat *wcb = cvCreateMat(wc->rows, wc->cols, CV_64FC1);
		cvSub(wc, wb, wcb); 
		double fwcb = Norm(wcb);

		//cvAdd(wx1, wcb, wx1);
		double distanceClosed = fwx1 + fwcb;//cvmGet(wx1, 0, 0);

		/*cvReleaseMat(&pImpactSound);*/
		cvReleaseMat(&hannWindow);
		cvReleaseMat(&pImpactSoundTr);
		cvReleaseMat(&impSoundTr);
		cvReleaseMat(&pOutput);
		cvReleaseMat(&freq);
		cvReleaseMat(&freqs);
		cvReleaseMat(&magnitudes);
		cvReleaseMat(&impactCepstrum);
		cvReleaseMat(&impCepstrumTr);
		cvReleaseMat(&wc);
		cvReleaseMat(&wx);
		cvReleaseMat(&wx0);
		cvReleaseMat(&wca);
		cvReleaseMat(&wx1);
		cvReleaseMat(&wcb);

		cvReleaseMat(&ukCTr);
		/// Karþýlaþtýr
		if(distanceOpen < distanceClosed)			
			openFistik++;
		else
			closedFistik++;
		
		return 1;

	}
	//delete pDataD;
	cvReleaseMat(&soundMatrix);
	return 0;
}

int SoundUtils::LoadFromImpactData(int numberOfTrainingSamples, double * pData, int dataSize)
{

	if (X.size()<numberOfTrainingSamples)
	{

		CvMat *pImpactSound;
		pImpactSound = cvCreateMat(1,288, CV_64FC1);
		*pImpactSound=cvMat(1,288, CV_64FC1, pData);

		/*PrintMatrix(pImpactSound);*/

		CvSize pImpactSize = cvGetSize(pImpactSound);
		if(pImpactSize.width>0)
		{

			CvMat *hannWindow = CalcHannWindow(pImpactSound->cols);

			CvMat* pImpactSoundTr = Transpose(pImpactSound);

			X_Data.push_back(Transpose(pImpactSound));

			cvMul(pImpactSoundTr, hannWindow, pImpactSoundTr);
			CvMat * pOutput = FFT(pImpactSoundTr, 256);

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

			Norm(magnitudes);
			Norm(pImpactSound);

			X.push_back(Transpose(pImpactSound));
			
			CvMat *impactCepstrum = MelCeptrum(magnitudes, freqs);
			//PrintMatrix(impactCepstrum);
			Norm(impactCepstrum);
			//PrintMatrix(impactCepstrum);
			C.push_back(Transpose(impactCepstrum));
		}

		// 10 düzgün ses olduðunda
		if (X.size() == 40)
		{
			XM = stdVector2CvMat(X);
			CM = stdVector2CvMat(C);
			XM_Data =stdVector2CvMat(X_Data);

			PrintMatrix(XM_Data);
			int numGoodInputs = findFeatureVectors(  );

			//PrintMatrix(w0);
			//PrintMatrix(wa);
			//PrintMatrix(w1);
			//PrintMatrix(wb);
			//PrintMatrix(ukX);//Vectors X
			//PrintMatrix(ukC);//Vectors C
			//PrintMatrix(Xm);// X PSI, mean
			//PrintMatrix(Cm);// C PSI Mean

			if (numGoodInputs < 10)
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

//class SoundUtils
//{
//	:
//	SoundUtils(void);
//	~SoundUtils(void);
//	CvMat* ExtractImpactSound(CvMat * pData, int nSamplingRate);
//	CvMat* CalcHannWindow(CvMat *pData, int nWindowSize);
//	double Norm(CvMat *Data);
//	CvMat* MelCeptrum(CvMat* magnitudes, CvMat *freqs );
//	CvMat* ExtractBands(CvMat* magnitudes, CvMat *freqs,
//		double startFreq, double breakPoint, double endFreq, int numberOfPieces);
//	CvMat* FindIndices(CvMat* x, CvMat* values );
//	CvMat* Energy(CvMat * x);
//	CvMat* Cepstrum(CvMat * e, int k);
//
//	void findFeatureVectors(std::vector<CvMat *> X, std::vector<CvMat *> C,
//		CvMat * w0, CvMat *wa, CvMat *w1, CvMat *wb, CvMat * ukX, CvMat *ukC, CvMat *Xm, CvMat *Cm) SoundUtils::findFeatureVectors( void ) ~SoundUtils(void)
//	{
//
//	}
