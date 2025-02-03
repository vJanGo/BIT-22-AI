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

//定义结构：进程PID、进程名称
struct ProcessInfo
{
    DWORD PID;
    string PName;
    ProcessInfo(DWORD PID, string PNmae) : PID(PID), PName(PNmae) {}

    //排序条件：PID 从小到大降序
    bool operator < (const ProcessInfo& rhs) const {
        return (PID < rhs.PID);
    }
};


// WCHAR 转换为 std::string
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


// 将 char* 转换为 LPWSTR
void ConvertToLPWSTR(const char* str, LPWSTR& outWStr) {
    int len = MultiByteToWideChar(CP_ACP, 0, str, -1, nullptr, 0);
    outWStr = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, str, -1, outWStr, len);
}

//将字节数转为字符串打印输出
inline void printStrFormatByte(const WCHAR* info, DWORDLONG bytes)
{
    TCHAR tmp[MAX_PATH];
    ZeroMemory(tmp, sizeof(tmp));
    StrFormatByteSize(bytes, tmp, MAX_PATH);
    wcout << info << tmp << endl;
    return;
}

//打印地址
inline void printAddress(const WCHAR* info, LPVOID addr)
{
    wcout << info << hex << setw(8) << addr << endl;
}

inline void printDword(const WCHAR* info, DWORDLONG dw)//将DWORD转为DWORDLONG
{
    cout << info;
    std::cout << dw << endl;
}

//查询系统配置信息
void getSystemInfo()
{
    SYSTEM_INFO si;
    ZeroMemory(&si, sizeof(si));
    GetNativeSystemInfo(&si);
    DWORD mem_size = (DWORD*)si.lpMaximumApplicationAddress - (DWORD*)si.lpMinimumApplicationAddress;
    std::cout << "===系统配置信息===" << "\n";
    std::cout << "处理器个数                " << si.dwNumberOfProcessors << std::endl;
    std::cout << "物理页大小                " << si.dwPageSize / 1024 << " KB" << std::endl;
    std::cout << "进程最小寻址空间：        0x" << std::hex << si.lpMinimumApplicationAddress << std::dec << std::endl;
    std::cout << "进程最大寻址地址:         0x" << std::hex << si.lpMaximumApplicationAddress << std::dec << std::endl;
    std::cout << "进程可用空间大小：        " << mem_size / 1024 / 1024 << " MB" << std::endl;
    return;
}

void ShowSystemMemoryInfo() {
    // 获取全局内存状态
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    std::cout << "===全局内存状态===" << "\n";
    if (GlobalMemoryStatusEx(&memStatus)) {
        std::cout << "总物理内存: " << memStatus.ullTotalPhys / (1024 * 1024) << " MB\n";
        std::cout << "可用物理内存: " << memStatus.ullAvailPhys / (1024 * 1024) << " MB\n";
        std::cout << "内存负载: " << memStatus.dwMemoryLoad << "%\n";
    }

    // 获取性能信息youguan 
    PERFORMANCE_INFORMATION perfInfo;
    perfInfo.cb = sizeof(perfInfo);
    if (GetPerformanceInfo(&perfInfo, sizeof(perfInfo))) {
        std::cout << "===性能信息===" << "\n";
        // 显示提交虚拟内存的相关信息
        std::cout << "已提交虚拟内存总量: " << (perfInfo.CommitTotal * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "虚拟内存上限: " << (perfInfo.CommitLimit * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "提交峰值: " << (perfInfo.CommitPeak * perfInfo.PageSize) / (1024 * 1024) << " MB\n";

        // 显示物理内存的相关信息
        std::cout << "总物理内存: " << (perfInfo.PhysicalTotal * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "可用物理内存: " << (perfInfo.PhysicalAvailable * perfInfo.PageSize) / (1024 * 1024) << " MB\n";

        // 显示内核内存的相关信息
        std::cout << "内核总内存: " << (perfInfo.KernelTotal * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "内核分页内存: " << (perfInfo.KernelPaged * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
        std::cout << "内核非分页内存: " << (perfInfo.KernelNonpaged * perfInfo.PageSize) / (1024 * 1024) << " MB\n";
    }
}

// 显示目标进程的虚拟地址空间
void ShowProcessMemoryInfo(DWORD processID) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
    if (!hProcess) {
        std::cerr << "无法打开进程: " << processID << "\n";
        return;
    }

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    MEMORY_BASIC_INFORMATION memInfo; //虚拟内存空间的基本信息结构
    LPVOID addr = sysInfo.lpMinimumApplicationAddress; //addr指向开始的地址

    std::cout << "目标进程虚拟地址空间布局:\n";
    while (addr < sysInfo.lpMaximumApplicationAddress) {
        if (VirtualQueryEx(hProcess, addr, &memInfo, sizeof(memInfo)) == sizeof(memInfo)) {

            std::cout << "基地址: " << memInfo.BaseAddress
                << "\t" 
                << "块大小: " << memInfo.RegionSize /(1024)
                << "KB\t"
                << "    状态: " << (memInfo.State == MEM_COMMIT ? "已提交" : "空闲")
                << "\t"
                << "\n";
        }
        addr = (LPVOID)((SIZE_T)addr + memInfo.RegionSize); //指向结束地址
    }

    CloseHandle(hProcess);
}

// 获取进程工作集信息
void ShowWorkingSet(DWORD processID) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
    if (!hProcess) {
        std::cerr << "无法打开进程: " << processID << "\n";
        return;
    }

    PSAPI_WORKING_SET_INFORMATION wsi;
    if (QueryWorkingSetEx(hProcess, &wsi, sizeof(wsi))) {
        std::cout << "工作集大小: " << wsi.NumberOfEntries << " 页\n";
    }
    else {
        std::cerr << "获取工作集信息失败。\n";
    }

    CloseHandle(hProcess);
}

//获取当前系统的所有进程PID
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

    //拍摄进程快照
    hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

    if (hSnapshot == INVALID_HANDLE_VALUE)
    {
        //快照拍摄失败
        return PInfo;
    }

    if (!Process32First(hSnapshot, &ps))
    {
        return PInfo;
    }

    //将进程PID、进程名称存到容器Vector中
    do
    {
        PInfo.emplace_back(ps.th32ProcessID, WCHAR2String(ps.szExeFile));

    } while (Process32Next(hSnapshot, &ps));

    //关闭快照句柄
    CloseHandle(hSnapshot);

    //排序
    sort(PInfo.begin(), PInfo.end());

    return PInfo;
}

int main() {

    // 显示系统配置信息
    getSystemInfo();
    // 显示系统内存信息
    ShowSystemMemoryInfo();

    // 打印所有进程的PID和名称
    vector<ProcessInfo> PInfo = GetProcessInfo();

    if (PInfo.size())
    {
        cout << "===所有进程的PID和名称===" << endl;
        cout << "编号\tPID" << endl;
        for (vector<DWORD>::size_type iter = 0; iter < PInfo.size(); iter++)
        {
            cout << iter << "\t\t" << dec << PInfo[iter].PID << "\t" << PInfo[iter].PName.c_str() << endl;
        }
    }
    else
    {
        cout << "未找到进程" << endl;
    }

    // 输入目标进程ID
    DWORD processID;
    std::cout << "请输入进程ID: ";
    std::cin >> processID;

    // 显示目标进程内存信息
    ShowProcessMemoryInfo(processID);

    // 显示目标进程的工作集信息
    ShowWorkingSet(processID);

    return 0;
}
