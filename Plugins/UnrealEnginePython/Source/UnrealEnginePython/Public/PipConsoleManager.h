// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "PipConsoleManager.generated.h"

/**
 * 
 */
UCLASS()
class UPipConsoleManager : public UCheatManager
{
	GENERATED_UCLASS_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FString DefaultCommand;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FString CmdDirectory;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FString Prepend;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
	FString Postpend;
	
	/* Command to execute pip from ue4 console*/
	UFUNCTION(Exec)
	void pip(FString Arg1, FString Arg2);


	UFUNCTION(Exec)
	void shell(FString Arg1, FString Arg2);
};
