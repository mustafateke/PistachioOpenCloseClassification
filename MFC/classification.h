#pragma once
#include "stdafx.h"
class Classification
{
public:
	Classification(void);
	~Classification(void);

	int Classify(char * pData, int dataSize);
};
