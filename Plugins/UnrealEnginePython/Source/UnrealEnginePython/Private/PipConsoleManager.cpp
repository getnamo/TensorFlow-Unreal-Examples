// Fill out your copyright notice in the Description page of Project Settings.

#include "UnrealEnginePythonPrivatePCH.h"
#include "IPluginManager.h"
#include "PipConsoleManager.h"

UPipConsoleManager::UPipConsoleManager(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	DefaultCommand = FString("dir");
	FString PluginPath = IPluginManager::Get().FindPlugin(TEXT("UnrealEnginePython"))->GetBaseDir();
	Prepend = FString("dir ");
	Postpend = FString("");
	FString FullPath = FPaths::ConvertRelativePathToFull(FPaths::Combine(*PluginPath, TEXT("Binaries"), TEXT("Win64")));
	CmdDirectory = Prepend + FullPath.Replace(TEXT("/"), TEXT("\\"));
}

//using windows api for now
#if PLATFORM_WINDOWS
#include "AllowWindowsPlatformTypes.h"
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

FString WindowsExecWithResult(const FString& cmd) {
	FString FinalCommand = cmd;
	char buffer[128];
	std::string result = "";
	std::shared_ptr<FILE> pipe(_popen(TCHAR_TO_ANSI(*FinalCommand), "r"), _pclose);

	//auto PC = UGameplayStatics::GetPlayerController(GEngine->GetWorld(), 0);
	if (!pipe)
	{
		UE_LOG(LogTemp, Log, TEXT("popen() failed!"));
		return FString("popen() failed!");
	}
	while (!feof(pipe.get()))
	{
		if (fgets(buffer, 128, pipe.get()) != NULL)
		{
			FString BufferString = FString(buffer);
			UE_LOG(LogTemp, Log, TEXT("buffer: %s"), *BufferString);
			result += buffer;
		}
	}
	return FString(result.c_str());
}

void UPipConsoleManager::pip(FString Arg1, FString Arg2)
{
	UE_LOG(LogTemp, Log, TEXT("you passed %s and %s"), *Arg1, *Arg2);
	

	/*TCHAR szCmdline[] = TEXT("child");
	PROCESS_INFORMATION piProcInfo;
	STARTUPINFO siStartInfo;
	BOOL bSuccess = FALSE;

	bSuccess = CreateProcess(NULL,
		szCmdline,     // command line 
		NULL,          // process security attributes 
		NULL,          // primary thread security attributes 
		TRUE,          // handles are inherited 
		0,             // creation flags 
		NULL,          // use parent's environment 
		NULL,          // use parent's current directory 
		&siStartInfo,  // STARTUPINFO pointer 
		&piProcInfo);  // receives PROCESS_INFORMATION */

	//NB: should be
	//Scripts\\pip.exe
	FString Result = WindowsExecWithResult(DefaultCommand);
	//UE_LOG(LogTemp, Log, TEXT("Result: %s"), *Result);
}

void UPipConsoleManager::shell(FString Arg1, FString Arg2)
{
	//prepend our bin directory
	FString Command = CmdDirectory + Arg1 + Postpend;

	if (!Arg2.IsEmpty())
	{
		Command += " " + Arg2;
	}

	FString Result = WindowsExecWithResult(Command);
}

#include "HideWindowsPlatformTypes.h"

#else

void UPipConsoleManager::pip(FString Arg1, FString Arg2)
{
	UE_LOG(LogTemp, Log, TEXT("platform not supported"));
}
void UPipConsoleManager::shell(FString Arg1, FString Arg2)
{
	UE_LOG(LogTemp, Log, TEXT("platform not supported"));
}



#endif