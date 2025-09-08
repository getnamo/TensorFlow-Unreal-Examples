#include "MLProcess.h"
#include "GenericPlatform/GenericPlatformProcess.h"

#if PLATFORM_WINDOWS
#include "Windows/WindowsHWrapper.h"
#endif

FMLProcess::FMLProcess()
{

}

FMLProcess::~FMLProcess()
{
	if (IsRunning())
	{
		Terminate();
	}
	Close();
}


TSharedPtr<FMLProcess> FMLProcess::Create(const FString& URL, const FString& Parms, bool bLaunchDetached, bool bLaunchHidden, bool bLaunchReallyHidden, int32 PriorityModifier, const FString& OptionalWorkingDirectory, bool bUsePipe)
{
	uint32 ProcessID;
	void* ReadPipe{ nullptr };
	void* WritePipe{ nullptr };
	if (bUsePipe)
	{
		FPlatformProcess::CreatePipe(ReadPipe, WritePipe);
	}
	auto Handle = FPlatformProcess::CreateProc(*URL, *Parms, bLaunchDetached, bLaunchHidden, bLaunchReallyHidden, &ProcessID, PriorityModifier, OptionalWorkingDirectory.Len() ? *OptionalWorkingDirectory : nullptr, WritePipe);
	if (Handle.IsValid())
	{
		TSharedPtr<FMLProcess> Instance = MakeShareable(new FMLProcess());
		Instance->ProcessHandle = Handle;
		Instance->ProcessID = ProcessID;
		Instance->ReadPipe = ReadPipe;
		Instance->WritePipe = WritePipe;
		return Instance;
	}
	else
	{
		return nullptr;
	}
}

TSharedPtr<FMLProcess> FMLProcess::Open(const FString& ProcName)
{
#if PLATFORM_WINDOWS
	FPlatformProcess::FProcEnumerator ProcEnumerator;

	while (ProcEnumerator.MoveNext())
	{
		auto Current = ProcEnumerator.GetCurrent();
		if (Current.GetName() == ProcName)
		{
			return Open_PID(Current.GetPID());
		}
	}
#endif
	return nullptr;
}

TSharedPtr<FMLProcess> FMLProcess::Open_PID(int32 ProcessId)
{
#if PLATFORM_WINDOWS
	auto Handle = FPlatformProcess::OpenProcess(ProcessId);
	if (Handle.IsValid())
	{
		TSharedPtr<FMLProcess> Instance = MakeShareable(new FMLProcess());
		Instance->ProcessHandle = Handle;
		Instance->ProcessID = ProcessId;
		/*Instance->ReadPipe = ReadPipe;
		Instance->WritePipe = WritePipe;*/
		return Instance;
	}
#endif
	return nullptr;
}

FString FMLProcess::GetApplicationName(int32 ProcessId)
{
	return FPlatformProcess::GetApplicationName(ProcessId);
}

int32 FMLProcess::GetCurrentProcessId()
{
	return (int32)FPlatformProcess::GetCurrentProcessId();
}

bool FMLProcess::IsApplicationRunning_PID(int32 ProcessId)
{
	return FPlatformProcess::IsApplicationRunning(ProcessId);
}

bool FMLProcess::IsApplicationRunning(const FString& ProcName)
{
	return FPlatformProcess::IsApplicationRunning(*ProcName);
}

bool FMLProcess::IsRunning()
{
	return FPlatformProcess::IsProcRunning(ProcessHandle);
}

void FMLProcess::Close()
{
	FPlatformProcess::CloseProc(ProcessHandle);

	if (ReadPipe || WritePipe)
	{
		FPlatformProcess::ClosePipe(ReadPipe, WritePipe);
		ReadPipe = WritePipe = nullptr;
	}
}
void FMLProcess::Sleep(float Seconds)
{
	FPlatformProcess::Sleep(Seconds);
}

FString FMLProcess::ReadFromPipe()
{
	return FPlatformProcess::ReadPipe(ReadPipe);
}

bool FMLProcess::ReadArrayFromPipe(TArray<uint8>& Array)
{
	return FPlatformProcess::ReadPipeToArray(ReadPipe, Array);
}

bool FMLProcess::WriteToPipe(const FString& Message, FString& OutWritten)
{
	return FPlatformProcess::WritePipe(WritePipe, Message, &OutWritten);
}

void FMLProcess::SimulateKeypress(int32 KeyEvent)
{
#if PLATFORM_WINDOWS
	INPUT input;
	WORD vkey = KeyEvent;
	input.type = INPUT_KEYBOARD;
	input.ki.time = 0;
	input.ki.dwExtraInfo = 0;
	input.ki.wVk = vkey;
	input.ki.dwFlags = KEYEVENTF_UNICODE;
	SendInput(1, &input, sizeof(INPUT));

	input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
	SendInput(1, &input, sizeof(INPUT));
#endif
}

void FMLProcess::Wait()
{
	return FPlatformProcess::WaitForProc(ProcessHandle);
}

void FMLProcess::Terminate(bool KillTree)
{
	return FPlatformProcess::TerminateProc(ProcessHandle, KillTree);
}

bool FMLProcess::GetReturnCode(int32& ReturnCode)
{
	return FPlatformProcess::GetProcReturnCode(ProcessHandle, &ReturnCode);
}

void FMLProcess::SetEnvironmentVar(const FString& VarName, const FString& VarValue)
{
	FPlatformMisc::SetEnvironmentVar(*VarName, *VarValue);
}

FString FMLProcess::GetEnvironmentVar(const FString& VarName)
{
	return FPlatformMisc::GetEnvironmentVariable(*VarName);
}

void FMLProcess::LaunchURL(const FString& URL, const FString& Parms, FString& Error)
{
	FPlatformProcess::LaunchURL(*URL, *Parms, &Error);
}

bool FMLProcess::CanLaunchURL(const FString& URL)
{
	return FPlatformProcess::CanLaunchURL(*URL);
}

FString FMLProcess::GetString(const FString& Key, bool bFlag)
{
	if (Key == TEXT("BaseDir")) return FPlatformProcess::BaseDir();
	if (Key == TEXT("UserDir")) return FPlatformProcess::UserDir();
	if (Key == TEXT("UserSettingsDir")) return FPlatformProcess::UserSettingsDir();
	if (Key == TEXT("UserTempDir")) return FPlatformProcess::UserTempDir();
	if (Key == TEXT("ApplicationSettingsDir")) return FPlatformProcess::ApplicationSettingsDir();
	if (Key == TEXT("ComputerName")) return FPlatformProcess::ComputerName();
	if (Key == TEXT("UserName")) return FPlatformProcess::UserName(bFlag);
	if (Key == TEXT("ShaderDir")) return FPlatformProcess::ShaderDir();
	if (Key == TEXT("CurrentWorkingDirectory")) return FPlatformProcess::GetCurrentWorkingDirectory();
	if (Key == TEXT("ShaderWorkingDir")) return FPlatformProcess::ShaderWorkingDir();
	if (Key == TEXT("ExecutableName")) return FPlatformProcess::ExecutableName(bFlag);
	if (Key == TEXT("ModulePrefix")) return FPlatformProcess::GetModulePrefix();
	if (Key == TEXT("ModuleExtension")) return FPlatformProcess::GetModuleExtension();
	if (Key == TEXT("BinariesSubdirectory")) return FPlatformProcess::GetBinariesSubdirectory();
	if (Key == TEXT("ModulesDirectory")) return FPlatformProcess::GetModulesDirectory();
	return FString();
}