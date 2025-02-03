#include <windows.h>
#include <psapi.h>
#include <iostream>
#include <vector>
#include <string>
#include<TlHelp32.h>
#include<vector>
#include<algorithm>
#include <iomanip>	 
#include <Windows.h> 
#include <shlwapi.h>
#pragma comment(lib,"Shlwapi.lib") 
#pragma comment(lib, "Psapi.Lib")
#pragma comment(lib,"Kernel32.lib")
using namespace std;

//����ṹ������PID����������
struct ProcessInfo
{
    DWORD PID;
    string PName;
    ProcessInfo(DWORD PID, string PNmae) : PID(PID), PName(PNmae) {}

    //����������PID ��С������
    bool operator < (const ProcessInfo& rhs) const {
        return (PID < rhs.PID);
    }
};


// WCHAR ת��Ϊ std::string
string WCHAR2String(LPCWSTR pwszSrc)
{
    int nLen = WideCharToMultiByte(CP_ACP, 0, pwszSrc, -1, NULL, 0, NULL, NULL);
    if (nLen <= 0)
        return std::string("");

    char* pszDst = new char[nLen];
    if (NULL == pszDst)
        return string("");

    WideCharToMultiByte(CP_ACP, 0, pwszSrc, -1, pszDst, nLen, NULL, NULL);
    pszDst[nLen - 1] = 0;

    std::string strTmp(pszDst);
    delete[] pszDst;

    return strTmp;
}


// �� char* ת��Ϊ LPWSTR
void ConvertToLPWSTR(const char* str, LPWSTR& outWStr) {
    int len = MultiByteToWideChar(CP_ACP, 0, str, -1, nullptr, 0);
    outWStr = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, str, -1, outWStr, len);
}

//���ֽ���תΪ�ַ�����ӡ���
inline void printStrFormatByte(const WCHAR* info, DWORDLONG bytes)
{
    TCHAR tmp[MAX_PATH];
    ZeroMemory(tmp, sizeof(tmp));
    StrFormatByteSize(bytes, tmp, MAX_PATH);
    wcout << info << tmp << endl;
    return;
}

//��ӡ��ַ
inline void printAddress(const WCHAR* info, LPVOID addr)
{
    wcout << info << hex << setw(8) << addr << endl;
}

inline void printDword(const WCHAR* info, DWORDLONG dw)//��DWORDתΪDWORDLONG
{
    cout << info;
    std::cout << dw << endl;
}

//��ѯϵͳ������Ϣ
void getSystemInfo()
{
    SYSTEM_INFO si;
    ZeroMemory(&si, sizeof(si));
    GetNativeSystemInfo(&si);
    DWORD mem_size = (DWORD*)si.lpMaximumApplicationAddress - (DWORD*)si.lpMinimumApplicationAddress;
    std::cout << "===ϵͳ������Ϣ===" << "\n";
    std::cout << "����������                " << si.dwNumberOfProcessors << std::endl;
    std::cout << "����ҳ��С                " << si.dwPageSize / 1024 << " KB" << std::endl;
    std::cout << "������СѰַ�ռ䣺        0x" << std::hex << si.lpMinimumApplicationAddress << std::dec << std::endl;
    std::cout << "�������Ѱַ��ַ:         0x" << std::hex << si.lpMaximumApplicationAddress << std::dec << std::endl;
    std::cout << "���̿��ÿռ��С��        " << mem_size / 1024 / 1024 << " MB" << std::endl;
    return;
}

void ShowSystemMemoryInfo() {
    // ��ȡȫ���ڴ�״̬
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    std::cout << "===ȫ���ڴ�״̬===" << "\n";
    if (GlobalMemoryStatusEx(&memStatus)) {
        std::cout << "�������ڴ�: " << memStatus.ullTotalPhys / (1024 * 1024) << " MB\n";
        std::cout << "���������ڴ�: " << memStatus.ullAvailPhys / (1024 * 1024) << " MB\n";
        std::cout << "�ڴ渺��: " << memStatus.dwMemoryLoad << "%\n";
    }

    // ��ȡ������Ϣyouguan 
    PERFORMANCE_INFORMATION perfInfo;
    perfInfo.cb = sizeof(perfInfo);
    if (GetPerformanceInfo(&perfInfo, sizeof(perfInfo))) {
        std::cout << "===������Ϣ===" << "\n";
        // ��ʾ�ύ�����ڴ�������Ϣ
        std::cout << "���ύ�����ڴ�����: " << (perfInfo.CommitTotal * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "�����ڴ�����: " << (perfInfo.CommitLimit * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "�ύ��ֵ: " << (perfInfo.CommitPeak * perfInfo.PageSize) / (1024 * 1024) << " MB\n";

        // ��ʾ�����ڴ�������Ϣ
        std::cout << "�������ڴ�: " << (perfInfo.PhysicalTotal * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "���������ڴ�: " << (perfInfo.PhysicalAvailable * perfInfo.PageSize) / (1024 * 1024) << " MB\n";

        // ��ʾ�ں��ڴ�������Ϣ
        std::cout << "�ں����ڴ�: " << (perfInfo.KernelTotal * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "�ں˷�ҳ�ڴ�: " << (perfInfo.KernelPaged * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "�ں˷Ƿ�ҳ�ڴ�: " << (perfInfo.KernelNonpaged * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
    }
}

// ��ʾĿ����̵������ַ�ռ�
void ShowProcessMemoryInfo(DWORD processID) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
    if (!hProcess) {
        std::cerr << "�޷��򿪽���: " << processID << "\n";
        return;
    }

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    MEMORY_BASIC_INFORMATION memInfo; //�����ڴ�ռ�Ļ�����Ϣ�ṹ
    LPVOID addr = sysInfo.lpMinimumApplicationAddress; //addrָ��ʼ�ĵ�ַ

    std::cout << "Ŀ����������ַ�ռ䲼��:\n";
    while (addr < sysInfo.lpMaximumApplicationAddress) {
        if (VirtualQueryEx(hProcess, addr, &memInfo, sizeof(memInfo)) == sizeof(memInfo)) {

            std::cout << "����ַ: " << memInfo.BaseAddress
                << "\t" 
                << "���С: " << memInfo.RegionSize /(1024)
                << "KB\t"
                << "    ״̬: " << (memInfo.State == MEM_COMMIT ? "���ύ" : "����")
                << "\t"
                << "\n";
        }
        addr = (LPVOID)((SIZE_T)addr + memInfo.RegionSize); //ָ�������ַ
    }

    CloseHandle(hProcess);
}

// ��ȡ���̹�������Ϣ
void ShowWorkingSet(DWORD processID) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
    if (!hProcess) {
        std::cerr << "�޷��򿪽���: " << processID << "\n";
        return;
    }

    PSAPI_WORKING_SET_INFORMATION wsi;
    if (QueryWorkingSetEx(hProcess, &wsi, sizeof(wsi))) {
        std::cout << "��������С: " << wsi.NumberOfEntries << " ҳ\n";
    }
    else {
        std::cerr << "��ȡ��������Ϣʧ�ܡ�\n";
    }

    CloseHandle(hProcess);
}

//��ȡ��ǰϵͳ�����н���PID
vector<ProcessInfo> GetProcessInfo()
{
    STARTUPINFO st;
    PROCESS_INFORMATION pi;
    PROCESSENTRY32 ps;
    HANDLE hSnapshot;
    vector<ProcessInfo> PInfo;

    ZeroMemory(&st, sizeof(STARTUPINFO));
    ZeroMemory(&pi, sizeof(PROCESS_INFORMATION));
    st.cb = sizeof(STARTUPINFO);
    ZeroMemory(&ps, sizeof(PROCESSENTRY32));
    ps.dwSize = sizeof(PROCESSENTRY32);

    //������̿���
    hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

    if (hSnapshot == INVALID_HANDLE_VALUE)
    {
        //��������ʧ��
        return PInfo;
    }

    if (!Process32First(hSnapshot, &ps))
    {
        return PInfo;
    }

    //������PID���������ƴ浽����Vector��
    do
    {
        PInfo.emplace_back(ps.th32ProcessID, WCHAR2String(ps.szExeFile));

    } while (Process32Next(hSnapshot, &ps));

    //�رտ��վ��
    CloseHandle(hSnapshot);

    //����
    sort(PInfo.begin(), PInfo.end());

    return PInfo;
}

int main() {

    // ��ʾϵͳ������Ϣ
    getSystemInfo();
    // ��ʾϵͳ�ڴ���Ϣ
    ShowSystemMemoryInfo();

    // ��ӡ���н��̵�PID������
    vector<ProcessInfo> PInfo = GetProcessInfo();

    if (PInfo.size())
    {
        cout << "===���н��̵�PID������===" << endl;
        cout << "���\tPID" << endl;
        for (vector<DWORD>::size_type iter = 0; iter < PInfo.size(); iter++)
        {
            cout << iter << "\t\t" << dec << PInfo[iter].PID << "\t" << PInfo[iter].PName.c_str() << endl;
        }
    }
    else
    {
        cout << "δ�ҵ�����" << endl;
    }

    // ����Ŀ�����ID
    DWORD processID;
    std::cout << "���������ID: ";
    std::cin >> processID;

    // ��ʾĿ������ڴ���Ϣ
    ShowProcessMemoryInfo(processID);

    // ��ʾĿ����̵Ĺ�������Ϣ
    ShowWorkingSet(processID);

    return 0;
}
