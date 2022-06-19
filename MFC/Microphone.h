#pragma once
 #include <mmsystem.h>
   class Microphone
   {
   public:

	   int InitialiseDevice(int samplesPerSec, int bitsPerSample);

	   int Open();

	   int Read( void* Dest, int numberOfSamples );

	   int Close();

	private :
		HWAVEIN      hWaveIn;
		WAVEFORMATEX pFormat;
   };

