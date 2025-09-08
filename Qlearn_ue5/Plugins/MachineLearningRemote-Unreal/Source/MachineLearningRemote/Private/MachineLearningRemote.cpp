// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "MachineLearningRemote.h"
#include "MachineLearningBase.h"

#if WITH_EDITOR
DEFINE_LOG_CATEGORY(MLBaseLog);
#endif

#define LOCTEXT_NAMESPACE "FMachineLearningRemoteModule"

void FMachineLearningRemoteModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
}

void FMachineLearningRemoteModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FMachineLearningRemoteModule, MachineLearningRemote)
