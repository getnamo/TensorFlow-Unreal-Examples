// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Runtime/Engine/Classes/Engine/LatentActionManager.h"
#include "Runtime/Json/Public/Dom/JsonValue.h"
#include "MachineLearningBaseComponent.generated.h"

/** Struct for combining data string and function for auto-serialized input */
USTRUCT()
struct FMLSendStringObject
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY()
	FString InputData;

	UPROPERTY()
	FString TargetFunction;
};


/** Wraps a float array in a UStruct for auto-serialization. May not be performant */
USTRUCT()
struct FMLSendRawObject
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY()
	TArray<float> InputData;

	UPROPERTY()
	FString TargetFunction;
};


DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FMLLogSignature, FString, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FMLConnectionSignature, FString, Endpoint);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FMLResultSignature, FString, ResultData, FString, CallingFunctionName);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FMLRawResultSignature, TArray<float>, ResultData, FString, CallingFunctionName);

/** 
*	Base component for both remote and standard MachineLearning components.
*	Considered abstract, ensures all MachineLearning components have same base API
*/
UCLASS(ClassGroup = Computing)
class MACHINELEARNINGBASE_API UMachineLearningBaseComponent : public UActorComponent
{
	GENERATED_BODY()

public:	

	UMachineLearningBaseComponent();

	/** 
	* When the given setup gets a connection to a backend (e.g. server for remote, library for native).
	* This is a good place to do initialization type calls instead of BeginPlay.
	*/
	UPROPERTY(BlueprintAssignable, Category = MachineLearningEvents)
	FMLConnectionSignature OnConnectedToBackend;

	UPROPERTY(BlueprintAssignable, Category = MachineLearningEvents)
	FMLConnectionSignature OnDisconnectedFromBackend;

	UPROPERTY(BlueprintAssignable, Category = MachineLearningEvents)
	FMLLogSignature OnLog;

	/** SendInput variant will return data to this event. */
	UPROPERTY(BlueprintAssignable, Category = MachineLearningEvents)
	FMLResultSignature OnInputResult;

	UPROPERTY(BlueprintAssignable, Category = MachineLearningEvents)
	FMLRawResultSignature OnRawInputResult;

	/** What this means is different for each sub-type */
	UPROPERTY(BlueprintReadOnly, Category = MachineLearningProperties)
	bool bIsConnectedToBackend;

	/** Send input to ML side result comes back to the OnResult event. Optionally re-target to another function name. */
	UFUNCTION(BlueprintCallable, Category = MachineLearningFunctions)
	virtual void SendStringInput(const FString& InputData, const FString& FunctionName = TEXT("onJsonInput"));

	/** Send float array input, bypasses encoding. Useful for large data/native inference, may not work in remote context. Result comes back to the OnRawResult event*/
	UFUNCTION(BlueprintCallable, Category = MachineLearningFunctions)
	virtual void SendRawInput(const TArray<float>& InputData, const FString& FunctionName = TEXT("onFloatArrayInput"));

	/** BP Only. Send input to ML side result will come back as a latent action in the graph. Recommended method. Optionally re-target to another function name. */
	UFUNCTION(BlueprintCallable, meta = (Latent, LatentInfo = "LatentInfo"), Category = MachineLearningFunctions)
	virtual void SendStringInputGraphCallback(const FString& InputData, FString& ResultData, struct FLatentActionInfo LatentInfo, const FString& FunctionName = TEXT("onJsonInput"));

	/** BP Only. Send float array input, bypasses encoding. Useful for large data/native inference, may not work in remote context. Result will come back as a latent action in the graph.*/
	UFUNCTION(BlueprintCallable, meta = (Latent, LatentInfo = "LatentInfo"), Category = MachineLearningFunctions)
	virtual void SendRawInputGraphCallback(const TArray<float>& InputData, TArray<float>& ResultData, struct FLatentActionInfo LatentInfo, const FString& FunctionName = TEXT("onFloatArrayInput"));

	//C++ Only. Native lambda callback variant, lambda must be specified to differentiate it from blueprint variant
	virtual void SendStringInput(const FString& InputData, TFunction<void(const FString& ResultData)> ResultCallback, const FString& FunctionName = TEXT("onJsonInput"));
	virtual void SendRawInput(const TArray<float>& InputData, TFunction<void(TArray<float>& ResultData)> ResultCallback, const FString& FunctionName = TEXT("onFloatArrayInput"));

	//convenience variant, to be added later
	//virtual void SendJsonInput(TSharedPtr<FJsonObject> InputData, TFunction<void(TSharedPtr<FJsonObject> ResultData)> ResultCallback, const FString& FunctionName = TEXT("onJsonInput"));

private:
	void ImmediateLatentResponse(struct FLatentActionInfo LatentInfo);

};
