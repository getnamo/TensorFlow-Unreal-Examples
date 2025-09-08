#pragma once

//Wrapper process class largely inspired from unreal.js's JavascriptProcess which is licensed under BSD 3-clause
class MLPROCESS_API FMLProcess
{
public:

	FMLProcess();
	~FMLProcess();

	FProcHandle ProcessHandle;
	uint32 ProcessID;
	void* ReadPipe;
	void* WritePipe;

	static TSharedPtr<FMLProcess> Create(const FString& URL, const FString& Parms, bool bLaunchDetached, bool bLaunchHidden, bool bLaunchReallyHidden, int32 PriorityModifier, const FString& OptionalWorkingDirectory, bool bUsePipe);

	static TSharedPtr<FMLProcess> Open(const FString& ProcName);

	static TSharedPtr<FMLProcess> Open_PID(int32 ProcessId);

	static FString GetApplicationName(int32 ProcessId);

	static int32 GetCurrentProcessId();

	static bool IsApplicationRunning_PID(int32 ProcessId);

	static bool IsApplicationRunning(const FString& ProcName);

	static void SetEnvironmentVar(const FString& VarName, const FString& VarValue);

	static FString GetEnvironmentVar(const FString& VarName);

	bool IsRunning();

	void Close();

	static void Sleep(float Seconds);

	void Wait();

	void Terminate(bool KillTree = false);

	bool GetReturnCode(int32& ReturnCode);

	FString ReadFromPipe();

	bool ReadArrayFromPipe(TArray<uint8>& Array);

	bool WriteToPipe(const FString& Message, FString& OutWritten);

	static void SimulateKeypress(int32 KeyEvent);

	static void LaunchURL(const FString& URL, const FString& Parms, FString& Error);

	static bool CanLaunchURL(const FString& URL);

	static FString GetString(const FString& Key, bool bFlag);
};