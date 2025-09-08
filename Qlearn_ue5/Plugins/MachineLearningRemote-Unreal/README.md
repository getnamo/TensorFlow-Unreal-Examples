# MachineLearningRemote Unreal Plugin
A Machine Learning (ML) plugin for the Unreal Engine, encapsulating calls to remote python servers running python ML libraries like Tensorflow or Pytorch. Depends on server complement repository: https://github.com/getnamo/ml-remote-server.

[![GitHub release](https://img.shields.io/github/release/getnamo/MachineLearningRemote-Unreal.svg)](https://github.com/getnamo/MachineLearningRemote-Unreal/releases)
[![Github All Releases](https://img.shields.io/github/downloads/getnamo/MachineLearningRemote-Unreal/total.svg)](https://github.com/getnamo/MachineLearningRemote-Unreal/releases)


Should have the same api as [tensorflow-ue4](https://github.com/getnamo/tensorflow-ue4), but with freedom to run a host server on platform of choice (e.g. remote win/linux/mac instances) and without a hard bind to the tensorflow library.

## Unreal Machine Learning Plugin Variants

Want to run tensorflow or pytorch on a remote (or local) python server?

- https://github.com/getnamo/MachineLearningRemote-Unreal

Want to use tensorflow python api with a python instance embedded in your unreal engine project? 

- https://github.com/getnamo/TensorFlow-Unreal

Want native tensorflow inference? (WIP)

- https://github.com/getnamo/TensorFlowNative-Unreal

## Quick Install & Setup

1. Install and setup https://github.com/getnamo/ml-remote-server on your target backend (can be a local folder), or setup the one embedded in plugin.
2. Download [Latest Release](https://github.com/getnamo/MachineLearningRemote-Unreal/releases)
3. Create new or choose project.
4. Browse to your project folder (typically found at Documents/Unreal Project/{Your Project Root})
5. Copy Plugins folder into your Project root.
6. Plugin should be now ready to use. Remember to startup your [server](https://github.com/getnamo/ml-remote-server) when using this plugin. 

## How to use

### Blueprint API

Add a ```MachineLearningRemote``` component to an actor of choice

![](https://i.imgur.com/Mx3gNAi.png)

Change server endpoint and ```DefaultScript``` to fit your use case. ```DefaultScript``` is the file name of your ML script which is placed in your *<[server](https://github.com/getnamo/ml-remote-server)>/scripts* folder. See https://github.com/getnamo/MachineLearningRemote-Unreal#python-api for example scripts.

![](https://i.imgur.com/R3YVPtm.png)

In your script the ```on_setup``` and if ```self.should_train_on_start``` is true ```on_begin_training``` gets called. When your script has trained or it is otherwise ready, you can send inputs to it using ```SendSIOJsonInput``` or other variants (string/raw).

![](https://i.imgur.com/WjmFLAu.png)

Your inputs will be processed on your script side and any value you return from there will be sent back and returned in ```ResultData``` as *USIOJsonValue* in your latent callback.

#### Other input variants

See https://github.com/getnamo/MachineLearningRemote-Unreal/blob/master/Source/MachineLearningRemote/Public/MachineLearningRemoteComponent.h for all variants

#### Custom Function

Change the ```FunctionName``` parameter in the ```SendSIOJsonInput``` to call a different function name in your script. This name will be used verbatim.

### Python API
These scripts should be placed in your *<[server](https://github.com/getnamo/ml-remote-server)>/scripts* folder. If a matching script is defined in your ```MachineLearningRemote```->```DefaultScript``` property it should load on connect.

Keep in mind that ```tensorflow``` is optional and used as an illustrative example of ML, you can use any other valid python library e.g. pytorch instead without issue.

See https://github.com/getnamo/ml-remote-server/tree/master/scripts for additional examples.
See https://github.com/getnamo/TensorFlow-Unreal#python-api for more detailed api examples.

#### empty_example

Bare bones API example.

```python
import tensorflow as tf
from mlpluginapi import MLPluginAPI

class ExampleAPI(MLPluginAPI):

	#optional api: setup your model for training
	def on_setup(self):
		pass
		
	#optional api: parse input object and return a result object, which will be converted to json for UE4
	def on_json_input(self, input):
		result = {}
		return result

	#optional api: start training your network
	def on_begin_training(self):
		pass


#NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
#required function to get our api
def get_api():
	#return CLASSNAME.getInstance()
	return ExampleAPI.get_instance()
```
#### add_example

Super basic example showing how to add using the tensorflow library.

```python
import tensorflow as tf
import unreal_engine as ue #for remote logging only, this is a proxy import to enable same functionality as local variants
from mlpluginapi import MLPluginAPI

class ExampleAPI(MLPluginAPI):

	#expected optional api: setup your model for training
	def on_setup(self):
		self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		self.a = tf.placeholder(tf.float32)
		self.b = tf.placeholder(tf.float32)

		#operation
		self.c = self.a + self.b

		ue.log('setup complete')
		pass
		
	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def on_json_input(self, json_input):
		
		ue.log(json_input)

		feed_dict = {self.a: json_input['a'], self.b: json_input['b']}

		raw_result = self.sess.run(self.c, feed_dict)

		ue.log('raw result: ' + str(raw_result))

		return {'c':raw_result.tolist()}

	#custom function to change the op
	def change_operation(self, type):
		if(type == '+'):
			self.c = self.a + self.b

		elif(type == '-'):
			self.c = self.a - self.b
		ue.log('operation changed to ' + type)


	#expected optional api: start training your network
	def on_begin_training(self):
		pass
    
#NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
#required function to get our api
def get_api():
	#return CLASSNAME.get_instance()
	return ExampleAPI.get_instance()
```

#### mnist_simple

One of the most basic ML examples using tensorflow to train a softmax mnist recognizer.

```python
#Converted to ue4 use from: https://www.tensorflow.org/get_started/mnist/beginners
#mnist_softmax.py: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import unreal_engine as ue
from mlpluginapi import MLPluginAPI

import operator

class MnistSimple(MLPluginAPI):
	
	#expected api: storedModel and session, json inputs
	def on_json_input(self, jsonInput):
		#expect an image struct in json format
		pixelarray = jsonInput['pixels']
		ue.log('image len: ' + str(len(pixelarray)))

		#embedd the input image pixels as 'x'
		feed_dict = {self.model['x']: [pixelarray]}

		result = self.sess.run(self.model['y'], feed_dict)

		#convert our raw result to a prediction
		index, value = max(enumerate(result[0]), key=operator.itemgetter(1))

		ue.log('max: ' + str(value) + 'at: ' + str(index))

		#set the prediction result in our json
		jsonInput['prediction'] = index

		return jsonInput

	#expected api: no params forwarded for training? TBC
	def on_begin_training(self):

		ue.log("starting mnist simple training")

		self.scripts_path = ue.get_content_dir() + "Scripts"
		self.data_dir = self.scripts_path + '/dataset/mnist'

		mnist = input_data.read_data_sets(self.data_dir)

		# Create the model
		x = tf.placeholder(tf.float32, [None, 784])
		W = tf.Variable(tf.zeros([784, 10]))
		b = tf.Variable(tf.zeros([10]))
		y = tf.matmul(x, W) + b

		# Define loss and optimizer
		y_ = tf.placeholder(tf.int64, [None])

		# The raw formulation of cross-entropy,
		#
		#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
		#                                 reduction_indices=[1]))
		#
		# can be numerically unstable.
		#
		# So here we use tf.losses.sparse_softmax_cross_entropy on the raw
		# outputs of 'y', and then average across the batch.
		cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

		#update session for this thread
		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		# Train
		for i in range(1000):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			if i % 100 == 0:
				ue.log(i)
				if(self.should_stop):
					ue.log('early break')
					break 

		# Test trained model
		correct_prediction = tf.equal(tf.argmax(y, 1), y_)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		finalAccuracy = self.sess.run(accuracy, feed_dict={x: mnist.test.images,
										  y_: mnist.test.labels})
		ue.log('final training accuracy: ' + str(finalAccuracy))
		
		#return trained model
		self.model = {'x':x, 'y':y, 'W':W,'b':b}

		#store optional summary information
		self.summary = {'x':str(x), 'y':str(y), 'W':str(W), 'b':str(b)}

		self.stored['summary'] = self.summary
		return self.stored

#required function to get our api
def get_api():
	#return CLASSNAME.get_instance()
	return MnistSimple.get_instance()
```


### C++ API

Available since 0.3.1. 

Same as blueprint API except for one additional callback variant. Use the lambda overloaded functions e.g. assuming you have a component defined as
```c++
UMachineLearningRemoteComponent* MLComponent; //NB: this needs to be allocated with NewObject or CreateDefaultSubobject 
```

#### SendRawInput
```c++
//Let's say you want to send some raw data
TArray<float> InputData;
//... fill

MLComponent->SendRawInput(InputData, [this](TArray<float>& ResultData)
{
	//Now we got our results back, do something with them here
}, FunctionName);
```

#### SendStringInput

Keep in mind that if you're using USIOJConvert utilities you'll need to add *SIOJson*, and *Json* as dependency modules in your project build.cs.

```c#
PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "Json", "SIOJson" });
```

Sending just a String

```c++
FString InputString = TEXT("Some Data");

MLComponent->SendStringInput(InputString, [this](const FString& ResultData)
{
	//e.g. just print the result
	UE_LOG(LogTemp, Log, TEXT("Got some results: %s"), *ResultData);
}, FunctionName);
```

A custom JsonObject

```c++
//Make an object {"myKey":"myValue"}
TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
JsonObject->SetStringField(TEXT("myKey"), TEXT("myValue"));
FString InputString = USIOJConvert::ToJsonString(JsonObject);

MLComponent->SendStringInput(InputString, [this](const FString& ResultData)
{
	//assuming you got a json string response we could query it, e.g. assume {"someNumber":5}
	TSharedPtr<FJsonObject> JsonObject = USIOJConvert::ToJsonObject(ResultData);
	double MyNumber = JsonObject->GetNumberField("someNumber");
	
	//do something with your number result
}, FunctionName);
```

Structs via Json
```c++
//Let's say you want to send some struct data in json format

USTRUCT()
struct FTestCppStruct
{
	GENERATED_BODY()

	UPROPERTY()
	int32 Index;

	UPROPERTY()
	float SomeNumber;

	UPROPERTY()
	FString Name;
};

//...

FTestCppStruct TestStruct;
TestStruct.Name = TEXT("George");
TestStruct.Index = 5;
TestStruct.SomeNumber = 5.123f;
FString StructJsonString = USIOJConvert::ToJsonString(USIOJConvert::ToJsonObject(FTestCppStruct::StaticStruct(), &TestStruct));

//In this example we're using the same struct type for the result, but you could use a different one or custom Json
FTestCppStruct ResultStruct;

MLComponent->SendStringInput(StructJsonString, [this, &ResultStruct](const FString& ResultData)
{
	//do something with the result, let's say we we have another struct of same type and we'd like to fill it with the results
	USIOJConvert::JsonObjectToUStruct(USIOJConvert::ToJsonObject(ResultData), FTestCppStruct::StaticStruct(), &ResultStruct);
}, FunctionName);
```
