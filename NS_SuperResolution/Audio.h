#pragma once

#include <Windows.h>
#include <mmsystem.h>

class WaveOut
{
public:
    WaveOut(PWAVEFORMATEX pWaveformat, int buf_ms = 80);
    ~WaveOut();

    void Start();

    void PlayAudio(char* buf, unsigned int nSize);

    void Stop();
private:
    char* buf1, * buf2;
    unsigned int buf_size;
    bool isplaying1, isplaying2;
    bool thread_is_running;
    HANDLE        m_hThread;
    DWORD        m_ThreadID;
    BOOL        m_bDevOpen;
    HWAVEOUT    m_hWaveOut;
    int            m_BufferQueue;
    WAVEFORMATEX m_Waveformat;
    WAVEHDR wavehdr1, wavehdr2;
    CRITICAL_SECTION m_Lock;

    static DWORD WINAPI ThreadProc(LPVOID lpParameter);
    void StartThread();
    void StopThread();
    void Open();

    //unprepare & close
    void Close();

    inline void WaitForPlayingEnd();

    inline void SetThreadSymbol(bool running);

    inline void SetFinishSymbol(PWAVEHDR pWaveHdr);
};

int audioStart();
int audioStop();
