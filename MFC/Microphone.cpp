#include "stdafx.h"
#include "Microphone.h"
   
  
	int Microphone::InitialiseDevice(int samplesPerSec, int bitsPerSample)
	{
		int failed = FALSE;

		this->pFormat.wFormatTag = WAVE_FORMAT_PCM;
		this->pFormat.nChannels = 1;						
		this->pFormat.nSamplesPerSec = samplesPerSec;
		this->pFormat.nAvgBytesPerSec = this->pFormat.nChannels * samplesPerSec * (bitsPerSample / 8);   // = nSamplesPerSec * n.Channels * wBitsPerSample/8
		this->pFormat.nBlockAlign = this->pFormat.nChannels * bitsPerSample / 8;	
		this->pFormat.wBitsPerSample = bitsPerSample;
		this->pFormat.cbSize = 0;

		return ! failed;
	}

	int Microphone::Open()
	{
		int failed = FALSE;

		MMRESULT result = waveInOpen(&this->hWaveIn, WAVE_MAPPER, &this->pFormat, 0L, 0L, WAVE_FORMAT_DIRECT);
		if (result)
		{
		  failed = TRUE;
		}

		return ! failed;
	}

	int Microphone::Read(void *Dest, int numberOfSamples)
	{
		int failed = FALSE;

		WAVEHDR      WaveInHdr;
		int NUMPTS = this->pFormat.wBitsPerSample * numberOfSamples / 8;

		// Set up and prepare header for input
		WaveInHdr.lpData = (LPSTR)Dest;
		WaveInHdr.dwBufferLength = NUMPTS;
		WaveInHdr.dwFlags = 0;

		MMRESULT result = waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WaveInHdr));
		if(result)
		{
			return 0;
		}

		// Insert a wave input buffer
		result = waveInAddBuffer(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
		if (result)
		{
			return 0;
		}

		result = waveInStart(hWaveIn);
		if (result)
		{
			return 0;
		}
		// Wait until finished recording
		do {} while (waveInUnprepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR)) == WAVERR_STILLPLAYING);
		
		return WaveInHdr.dwBytesRecorded;
	}

	int Microphone::Close()
	{
		int failed = FALSE;

		MMRESULT result = waveInClose(hWaveIn);
		if(result)
		{
			failed = TRUE;
		}

		return ! failed;
	}

