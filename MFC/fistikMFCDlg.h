// fistikMFCDlg.h : header file
//

#pragma once
#include "SoundUtils.h"
#include "Microphone.h"
#include <time.h>
// CfistikMFCDlg dialog
class CfistikMFCDlg : public CDialog
{
// Construction
public:
	CfistikMFCDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_FISTIKMFC_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;
	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnBnClickedTrain();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnEnChangeEdit1();
	CStatic *lblTraining;
	static CStatic *lblClassification;
	static SoundUtils sound;
	static Microphone mic; // Sürekli okuduðu verileri almasý için, fistiðin sesin sonunda düþmesi durumunda sonraki sesin deðerini bufferdan almak için static olmalý
	static bool bClassify;

	afx_msg void OnBnClickedClassify();
	afx_msg void OnBnClickedReset();
	afx_msg void OnBnClickedLoad();
	afx_msg void OnBnClickedSave();



	static UINT __cdecl ClassifyThread(LPVOID pParam) 
	{
		mic.Open();
		while (true)
		{
			if (bClassify)
			{
				//Microphone mic;

				//mic.InitialiseDevice(samplingRate, 16);


				time_t Start_t, End_t; 
				int time_task1;

				Start_t = time(NULL);   

			
				
				int nSamples =samplingRate/nFistikPerSec;

				short *pData = new short[nSamples];


				mic.Read(pData, nSamples );

				if (sound.Classify(pData, nSamples))
				{
					End_t = time(NULL);    //record time that task 1 ends 
					time_task1 = difftime(End_t, Start_t);    //compute elapsed time of task 1	
					CString str;
					str.Format( "Açýk:\t %d\tKapalý:\t %d", sound.openFistik, sound.closedFistik/*,time_task1 */);
					lblClassification->SetWindowText(str);
				}
				
				delete pData;
			}

			Sleep(5);
		}
		mic.Close();
		return 0;
	}


	afx_msg void OnBnClickedButtonSelfTest();
};

