#include "Audio.h"
#include <string>
#include <iostream>
#include <windows.h>
#pragma comment(lib,"winmm.lib")

//#define _PRINT
#include <stdexcept>

using namespace std;

const int NUM_BUFFERS = 5;
const int BUFFER_SIZE = 14112;

HWAVEOUT hWaveOut;
HWAVEIN hWaveIn;
WAVEHDR waveBlocks[NUM_BUFFERS];
CRITICAL_SECTION waveCriticalSection;
WaveOut* waveOut;

void CALLBACK waveInCallback(HWAVEIN hwi, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
    if (uMsg != WIM_DATA) return;

    //EnterCriticalSection(&waveCriticalSection);


    WAVEHDR* p = (WAVEHDR*)dwParam1;
    waveOut->PlayAudio(p->lpData, p->dwBytesRecorded);


    HRESULT hr = waveInAddBuffer(hWaveIn, (WAVEHDR*)dwParam1, sizeof(WAVEHDR));
    if (hr != MMSYSERR_NOERROR)
        std::cout << hr;
    //LeaveCriticalSection(&waveCriticalSection);
}

int GetAudioInDeviceId()
{
    UINT numDevices = waveInGetNumDevs();
    if (numDevices == 0) {
        std::cout << "No AudioIn device found." << std::endl;
        return -1;
    }
    std::cout << "Found AudioIn device:" << std::endl;
    for (UINT i = 0; i < numDevices; ++i) {
        WAVEINCAPS waveInCaps;
        MMRESULT result = waveInGetDevCaps(i, &waveInCaps, sizeof(WAVEINCAPS));
        if (result == MMSYSERR_NOERROR) {
            std::cout << "Device ID: " << i << std::endl;//<< ", name: " << waveInCaps.szPname << std::endl;
        }
        else {
            std::cout << "Cannot get info of device " << i << std::endl;
        }
    }

    std::cout << "Please select an AudioIn device ID: ";
    UINT choice;
    std::cin >> choice;

    if (choice >= numDevices) {
        std::cout << "Invalid input" << std::endl;
        return -1;
    }

    return 0;
}

int audioStart()
{
    InitializeCriticalSection(&waveCriticalSection);

    WAVEFORMATEX waveFormat;
    waveFormat.wFormatTag = WAVE_FORMAT_PCM;
    waveFormat.nChannels = 2;
    waveFormat.nSamplesPerSec = 44100;
    waveFormat.wBitsPerSample = 16;
    waveFormat.nBlockAlign = (waveFormat.wBitsPerSample * waveFormat.nChannels) >> 3;
    waveFormat.nAvgBytesPerSec = waveFormat.nBlockAlign * waveFormat.nSamplesPerSec;
    waveFormat.cbSize = 0;

    waveOut = new WaveOut(&waveFormat);
    waveOut->Start();

    if (waveOutOpen(&hWaveOut, WAVE_MAPPER, &waveFormat, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR)
    {
        std::cerr << "Error opening output device!" << std::endl;
        return 1;
    }

    UINT deviceId = GetAudioInDeviceId();
    if (waveInOpen(&hWaveIn, deviceId, &waveFormat, (DWORD_PTR)waveInCallback, 0, CALLBACK_FUNCTION) != MMSYSERR_NOERROR)
    {
        std::cerr << "Error opening input device!" << std::endl;
        return 1;
    }

    for (int i = 0; i < NUM_BUFFERS; ++i)
    {
        char* buffer = new char[BUFFER_SIZE];
        waveBlocks[i].dwBufferLength = BUFFER_SIZE;
        waveBlocks[i].dwBytesRecorded = 0;
        waveBlocks[i].lpData = buffer;
        waveInPrepareHeader(hWaveIn, &waveBlocks[i], sizeof(WAVEHDR));
        waveInAddBuffer(hWaveIn, &waveBlocks[i], sizeof(WAVEHDR));
    }

    if (waveInStart(hWaveIn) != MMSYSERR_NOERROR)
    {
        std::cerr << "Error starting recording!" << std::endl;
        return 1;
    }
    return 0;
}

int audioStop() {

    waveInStop(hWaveIn);

    for (int i = 0; i < NUM_BUFFERS; ++i)
    {
        waveInUnprepareHeader(hWaveIn, &waveBlocks[i], sizeof(WAVEHDR));
        waveOutUnprepareHeader(hWaveOut, &waveBlocks[i], sizeof(WAVEHDR));
        delete[] waveBlocks[i].lpData;
    }

    waveInClose(hWaveIn);
    waveOutClose(hWaveOut);

    DeleteCriticalSection(&waveCriticalSection);
    waveOut->Stop();
    return 0;
}

WaveOut::WaveOut(PWAVEFORMATEX pWaveformat, int buf_ms) :
    thread_is_running(false), m_hThread(0), m_ThreadID(0), m_bDevOpen(false), m_hWaveOut(0), m_BufferQueue(0), isplaying1(false), isplaying2(false)
{
    memcpy(&m_Waveformat, pWaveformat, sizeof(WAVEFORMATEX));
    m_Waveformat.nBlockAlign = (m_Waveformat.wBitsPerSample * m_Waveformat.nChannels) >> 3;
    m_Waveformat.nAvgBytesPerSec = m_Waveformat.nBlockAlign * m_Waveformat.nSamplesPerSec;

    buf_size = buf_ms * m_Waveformat.nSamplesPerSec * m_Waveformat.nBlockAlign / 1000;
    buf1 = new char[buf_size];

    buf2 = new char[buf_size];

    std::cout << buf_size << std::endl;

    ZeroMemory(&wavehdr1, sizeof(WAVEHDR));
    ZeroMemory(&wavehdr2, sizeof(WAVEHDR));

    wavehdr1.lpData = buf1;
    wavehdr1.dwBufferLength = buf_size;

    wavehdr2.lpData = buf2;
    wavehdr2.dwBufferLength = buf_size;

    InitializeCriticalSection(&m_Lock);

}

WaveOut::~WaveOut()
{
    WaitForPlayingEnd();
    StopThread();
    Close();
    delete[] buf1;
    delete[] buf2;
    DeleteCriticalSection(&m_Lock);
}

void WaveOut::Start()
{
    StartThread();
    try
    {
        Open();
    }
    catch (runtime_error e)
    {
        StopThread();
        throw e;
    }
}

void WaveOut::PlayAudio(char* in_buf, unsigned int in_size)
{
    if (!m_bDevOpen)
    {
        throw runtime_error("waveOut has not been opened");
    }

    while (1)
    {
        if (isplaying1 && isplaying2)
        {
            Sleep(10);
#ifdef _PRINT
            printf("PlayAudio::waitting\n");
#endif
            continue;
        }
        else
        {
#ifdef _PRINT
            printf("PlayAudio::break\n");
#endif
            break;
        }
    }

    //将没有在播放的hdr设为当前hdr
    char* now_buf = nullptr;
    WAVEHDR* now_wavehdr = nullptr;
    bool* now_playing = nullptr;
    if (isplaying1 == false)
    {
        now_buf = buf1;
        now_wavehdr = &wavehdr1;
        now_playing = &isplaying1;
    }

    if (isplaying2 == false)
    {
        now_buf = buf2;
        now_wavehdr = &wavehdr2;
        now_playing = &isplaying2;
    }

    if (in_size > buf_size)
    {
        throw runtime_error("input buffer size is bigger than self");
    }

    if (in_size <= buf_size)
    {
        now_wavehdr->dwBufferLength = in_size;
    }

    memcpy(now_buf, in_buf, in_size);

    if (waveOutWrite(m_hWaveOut, now_wavehdr, sizeof(WAVEHDR)) != MMSYSERR_NOERROR)
    {
        throw runtime_error("waveOutWrite fail");
    }
    EnterCriticalSection(&m_Lock);
    *now_playing = true;
    LeaveCriticalSection(&m_Lock);


}

DWORD __stdcall WaveOut::ThreadProc(LPVOID lpParameter)
{
#ifdef _PRINT
    printf("ThreadProc::enter\n");
#endif
    WaveOut* pWaveOut = (WaveOut*)lpParameter;
    pWaveOut->SetThreadSymbol(true);

    MSG msg;
    while (GetMessage(&msg, 0, 0, 0))
    {
        switch (msg.message)
        {
        case WOM_OPEN:
            break;
        case WOM_CLOSE:
            break;
        case WOM_DONE:
            WAVEHDR* pWaveHdr = (WAVEHDR*)msg.lParam;
            pWaveOut->SetFinishSymbol(pWaveHdr);
            break;
        }
    }
    pWaveOut->SetThreadSymbol(false);
#ifdef _PRINT
    printf("ThreadProc::exit\n");
#endif
    return msg.wParam;
}

void WaveOut::StartThread()
{
    if (thread_is_running)
    {
        throw runtime_error("thread has been running");
    }

    m_hThread = CreateThread(0, 0, ThreadProc, this, 0, &m_ThreadID);

    if (!m_hThread)
    {
        throw runtime_error("CreateThread fail");
    }
}

void WaveOut::StopThread()
{
    if (!thread_is_running)
    {
        return;
    }

    if (m_hThread)
    {
        PostThreadMessage(m_ThreadID, WM_QUIT, 0, 0);
        while (1)
        {
            if (thread_is_running)
            {
#ifdef _PRINT
                printf("StopThread::waiting\n");
#endif
                Sleep(1);
            }
            else
            {
#ifdef _PRINT
                printf("StopThread::break\n");
#endif
                break;
            }
        }
        TerminateThread(m_hThread, 0);
        m_hThread = 0;
    }
}

void WaveOut::Open()
{
    if (m_bDevOpen)
    {
        throw runtime_error("waveOut has been opened");
    }
    MMRESULT mRet;
    mRet = waveOutOpen(0, WAVE_MAPPER, &m_Waveformat, 0, 0, WAVE_FORMAT_QUERY);
    if (mRet != MMSYSERR_NOERROR)
    {
        throw runtime_error("waveOutOpen fail");
    }

    mRet = waveOutOpen(&m_hWaveOut, WAVE_MAPPER, &m_Waveformat, m_ThreadID, 0, CALLBACK_THREAD);
    if (mRet != MMSYSERR_NOERROR)
    {
        throw runtime_error("waveOutOpen fail");
    }

    if (waveOutPrepareHeader(m_hWaveOut, &wavehdr1, sizeof(WAVEHDR)) != MMSYSERR_NOERROR)
    {
        throw runtime_error("waveOutPrepareHeader fail");
    }

    if (waveOutPrepareHeader(m_hWaveOut, &wavehdr2, sizeof(WAVEHDR)) != MMSYSERR_NOERROR)
    {
        throw runtime_error("waveOutPrepareHeader fail");
    }

    m_bDevOpen = TRUE;
}

void WaveOut::Close()
{
    if (!m_bDevOpen)
    {
        return;
    }

    if (!m_hWaveOut)
    {
        return;
    }

    MMRESULT mRet;
    if ((mRet = waveOutUnprepareHeader(m_hWaveOut, &wavehdr1, sizeof(WAVEHDR))) != MMSYSERR_NOERROR)
    {
        TCHAR info[260];
        waveOutGetErrorText(mRet, info, 260);
        throw runtime_error("asf");
    }

    if ((mRet = waveOutUnprepareHeader(m_hWaveOut, &wavehdr2, sizeof(WAVEHDR))) != MMSYSERR_NOERROR)
    {
        TCHAR info[260];
        waveOutGetErrorText(mRet, info, 260);
        throw runtime_error("as");
    }

    mRet = waveOutClose(m_hWaveOut);
    if (mRet != MMSYSERR_NOERROR)
    {
        throw runtime_error("waveOutClose fail");
    }
    m_hWaveOut = 0;
    m_bDevOpen = FALSE;
}

inline void WaveOut::WaitForPlayingEnd()
{
    while (1)
    {
        if (isplaying1 || isplaying2)
        {
#ifdef _PRINT
            printf("Stop::waitting\n");
#endif
            Sleep(1);
        }
        else
        {
#ifdef _PRINT
            printf("Stop::break\n");
#endif
            break;
        }
    }
}

void WaveOut::Stop()
{
    MMRESULT mRet;
    if ((mRet = waveOutReset(m_hWaveOut)) != MMSYSERR_NOERROR)
    {
        TCHAR info[260];
        waveOutGetErrorText(mRet, info, 260);
        throw runtime_error("unk");
    }

    WaitForPlayingEnd();

    StopThread();

    Close();
}

inline void WaveOut::SetThreadSymbol(bool running)
{
    EnterCriticalSection(&m_Lock);
    thread_is_running = running;
    LeaveCriticalSection(&m_Lock);
}

inline void WaveOut::SetFinishSymbol(PWAVEHDR pWaveHdr)
{
    EnterCriticalSection(&m_Lock);
    if (pWaveHdr == &wavehdr1)
    {
        isplaying1 = false;
#ifdef _PRINT
        printf("1 is finished.\n");
#endif
    }
    else
    {
        isplaying2 = false;
#ifdef _PRINT
        printf("2 is finished.\n");
#endif
    }
    LeaveCriticalSection(&m_Lock);
}
