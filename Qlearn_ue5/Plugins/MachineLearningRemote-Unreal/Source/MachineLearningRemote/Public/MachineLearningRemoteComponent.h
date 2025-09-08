// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "MachineLearningBaseComponent.h"
#include "SocketIONative.h"
#include "SIOJsonValue.h"
#include "MLProcess.h"
#include "MachineLearningRemoteComponent.generated.h"

UENUM(BlueprintType)
enum class ETFServerType : uint8
{
	SERVER_PYTHON,
	SERVER_NODEJS
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FMLScriptSignature, FString, ScriptName);

/**
 * Implements ML API via remote server calls
 */
UCLASS(BlueprintType, Blueprintable, ClassGroup = Computing, meta = (BlueprintSpawnableComponent))
class MACHINELEARNINGREMOTE_API UMachineLearningRemoteComponent : public UMachineLearningBaseComponent
{
	GENERATED_BODY()
public:

	UMachineLearningRemoteComponent();
	~UMachineLearningRemoteComponent();

	UPROPERTY(BlueprintAssignable, Category = MachineLearningEvents)
	FMLScriptSignature OnScriptStarted;

	/** remote server and address, default localhost:3000 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	FString ServerAddressAndPort;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	FString SendInputEventName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	FString StartScriptEventName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	FString ScriptStartedEventName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	FString LogEventName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	bool bConnectOnBeginPlay;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	bool bStartScriptOnConnection;

	UPROPERTY(BlueprintReadOnly, Category = MLRemoteProperties)
	bool bScriptRunning;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	FString DefaultScript;

	/** Support both python and nodejs servers */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLRemoteProperties)
	ETFServerType ServerType;

	/** If true it will bootup a server on beginplay. Requires python to be installed. */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLEmbeddedProperties)
	bool bAutoStartServer;

	static bool bServerIsRunning;

	/** EXPERIMENTAL: If true, will launch an embedded server. Requires the embedded server type exists in third party. */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLEmbeddedProperties)
	bool bUseEmbeddedServer;

	/** EXPERIMENTAL: Relative to plugin root */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLEmbeddedProperties)
	FString EmbeddedServerRelativePath;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = MLEmbeddedProperties)
	bool bPrintServerLog;


	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, enum ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	/** For remote ML components we can use socket.io protocol to communicate objects directly. Return result in graph context. */
	UFUNCTION(BlueprintCallable, meta = (Latent, LatentInfo = "LatentInfo"), Category = MLFunctions)
	virtual void SendSIOJsonInput(USIOJsonValue* InputData, USIOJsonValue*& ResultData, struct FLatentActionInfo LatentInfo, const FString& FunctionName = TEXT("on_json_input"));

	/** Start designated script. Get's called on beginplay by default*/
	UFUNCTION(BlueprintCallable, Category = MLFunctions)
	virtual void StartScript(const FString& ScriptName);

	//NB by default python pep expects lowercase_lowercase function names. This is different from BaseComponent defaults, which uses camelCase
	virtual void SendStringInput(const FString& InputData, const FString& FunctionName = TEXT("on_json_input")) override;
	virtual void SendStringInput(const FString& InputData, TFunction<void(const FString& ResultData)> ResultCallback, const FString& FunctionName = TEXT("on_json_input")) override;
	virtual void SendRawInput(const TArray<float>& InputData, const FString& FunctionName = TEXT("on_float_array_input")) override;
	virtual void SendRawInput(const TArray<float>& InputData, TFunction<void(TArray<float>& ResultData)> ResultCallback, const FString& FunctionName = TEXT("on_float_array_input")) override;
	virtual void SendStringInputGraphCallback(const FString& InputData, FString& ResultData, struct FLatentActionInfo LatentInfo, const FString& FunctionName = TEXT("on_json_input")) override;
	virtual void SendRawInputGraphCallback(const TArray<float>& InputData, TArray<float>& ResultData, struct FLatentActionInfo LatentInfo, const FString& FunctionName = TEXT("on_float_array_input")) override;

protected:
	TSharedPtr<FSocketIONative> Socket;
	static TSharedPtr<FMLProcess> Process;
};
