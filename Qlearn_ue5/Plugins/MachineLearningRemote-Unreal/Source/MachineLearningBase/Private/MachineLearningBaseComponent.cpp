// Fill out your copyright notice in the Description page of Project Settings.

#include "MachineLearningBaseComponent.h"
#include "MachineLearningBase.h"
#include "Engine/Engine.h"
#include "Runtime/Engine/Public/LatentActions.h"

UMachineLearningBaseComponent::UMachineLearningBaseComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
	bIsConnectedToBackend = false;
}

void UMachineLearningBaseComponent::SendStringInput(const FString& InputData, const FString& FunctionName /*= TEXT("Default")*/)
{
	FString ResultData = TEXT("SendInput not implemented");
	UE_LOG(MLBaseLog, Warning, TEXT("%s"), *ResultData);
	OnInputResult.Broadcast(ResultData, FunctionName);
}

void UMachineLearningBaseComponent::SendStringInput(const FString& InputData, TFunction<void(const FString& ResultData)> ResultCallback, const FString& FunctionName /*= TEXT("onJsonInput")*/)
{
	UE_LOG(MLBaseLog, Warning, TEXT("SendStringInputLambdaOverload not implemented"));
	if (ResultCallback)
	{
		ResultCallback(TEXT(""));
	}
}

void UMachineLearningBaseComponent::SendRawInput(const TArray<float>& InputData, const FString& FunctionName /*= TEXT("onFloatArrayInput")*/)
{
	UE_LOG(MLBaseLog, Warning, TEXT("SendRawInput not implemented"));

	TArray<float> ResultData;
	OnRawInputResult.Broadcast(ResultData, FunctionName);
}

void UMachineLearningBaseComponent::SendRawInput(const TArray<float>& InputData, TFunction<void(TArray<float>& ResultData)> ResultCallback /*= nullptr*/, const FString& FunctionName /*= TEXT("onFloatArrayInput")*/)
{
	UE_LOG(MLBaseLog, Warning, TEXT("SendRawInputLambdaOverload not implemented"));
	if (ResultCallback)
	{
		TArray<float> ResultData;
		ResultCallback(ResultData);
	}
}

void UMachineLearningBaseComponent::SendStringInputGraphCallback(const FString& InputData, FString& ResultData, struct FLatentActionInfo LatentInfo, const FString& FunctionName /*= TEXT("Default")*/)
{
	ResultData = TEXT("SendInputGraphResult not implemented");

	UE_LOG(MLBaseLog, Warning, TEXT("%s"), *ResultData);

	ImmediateLatentResponse(LatentInfo);
}

void UMachineLearningBaseComponent::SendRawInputGraphCallback(const TArray<float>& InputData, TArray<float>& ResultData, struct FLatentActionInfo LatentInfo, const FString& FunctionName /*= TEXT("onJsonInput")*/)
{
	UE_LOG(MLBaseLog, Warning, TEXT("SendRawInputGraphResult not implemented"));
	ImmediateLatentResponse(LatentInfo);
}

void UMachineLearningBaseComponent::ImmediateLatentResponse(struct FLatentActionInfo LatentInfo)
{
	//Create a dummy pending latent action that returns in next tick
	UWorld* World = GEngine->GetWorldFromContextObject(this, EGetWorldErrorMode::LogAndReturnNull);
	if (!World)
	{
		return;
	}

	FLatentActionManager& LatentActionManager = World->GetLatentActionManager();
	FPendingLatentAction *LatentAction = LatentActionManager.FindExistingAction<FPendingLatentAction>(LatentInfo.CallbackTarget, LatentInfo.UUID);
	LatentAction = new FPendingLatentAction();	//safe to use new since latentactionmanager will delete it
	int32 UUID = LatentInfo.UUID;
	LatentActionManager.AddNewAction(LatentInfo.CallbackTarget, LatentInfo.UUID, LatentAction);
}