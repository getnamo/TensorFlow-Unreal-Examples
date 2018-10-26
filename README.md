# tensorflow-ue4-examples

[![GitHub release](https://img.shields.io/github/release/getnamo/tensorflow-ue4-examples/all.svg)](https://github.com/getnamo/tensorflow-ue4-examples/releases)
[![Github All Releases](https://img.shields.io/github/downloads/getnamo/tensorflow-ue4-examples/total.svg)](https://github.com/getnamo/tensorflow-ue4-examples/releases)

Example content project for [tensorflow-ue4](https://github.com/getnamo/tensorflow-ue4) plugin.

This repository also tracks changes required across all dependencies to make tensorflow work well with UE4.

See [issues](https://github.com/getnamo/tensorflow-ue4-examples/issues) for current work and bug reports.

[Unreal Forum Thread](https://forums.unrealengine.com/community/work-in-progress/1357673-tensorflow)

## Setup

 1.	(GPU only) [Install CUDA and cudNN pre-requisites](https://www.tensorflow.org/install/install_windows) if you're using compatible GPUs (NVIDIA)
 2. [Download latest project release](https://github.com/getnamo/tensorflow-ue4-examples/releases)
 3.	Download the matching tensorflow plugin release. Choose CPU download (or GPU version if hardware is supported). The matching plugin link is usually found under the [project release](https://github.com/getnamo/tensorflow-ue4-examples/releases).
 4.	Browse to your extracted project folder
 5. Copy *Plugins* folder from your plugin download into your Project root.
 6. Launch and wait for tensorflow dependencies to be installed. The tensorflow plugin will auto-resolve any dependencies listed in [Plugins/tensorflow-ue4/Content/Scripts/upymodule.json](https://github.com/getnamo/tensorflow-ue4/blob/master/Content/Scripts/upymodule.json) using pip. Note that this step may take a few minutes and depends on your internet connection speed and you will see nothing change in the output log window until the process has completed.
 
![image](https://user-images.githubusercontent.com/542365/36981363-e88aa2ec-2084-11e8-828c-e5a526cda67b.png)

 7. Once you see an output similar to the above in your console window, everything should be ready to go, try different examples from e.g. [Content/ExampleAssets/Maps](https://github.com/getnamo/tensorflow-ue4-examples/tree/master/Content/ExampleAssets/Maps)!
 
### Note on cloning the repository

If you're not using a release, but instead wish to clone the repository using git. Ensure you follow [TensorFlow-ue4 instructions on cloning](https://github.com/getnamo/tensorflow-ue4#note-on-git-clone).

## Examples

### Mnist recognition

Map is found under [_Content/ExampleAssets/Maps/Mnist.umap_](https://github.com/getnamo/tensorflow-ue4-examples/blob/master/Content/ExampleAssets/Maps/Mnist.umap)  and it should be the default map when the project launches.

[![mnist spawn samples](http://i.imgur.com/kvsLXvF.gif)](https://github.com/getnamo/tensorflow-ue4-examples/blob/master/Content/Scripts/mnistSpawnSamples.py)

*Default mnist example script: [mnistSpawnSamples.py](https://github.com/getnamo/tensorflow-ue4-examples/blob/master/Content/Scripts/mnistSpawnSamples.py)*

On map launch you'll have a basic example ready for play in editor. It should automatically train a really basic network when you hit play and then be ready to use in a few seconds. You can then press 'F' to send e.g. image of a 2 to predict. Press 0-9 numbers on your keyboard to change the input, press F again to send this updated input to classifier. Note that this is a very basic classifier and it will struggle to classify digits above 4 in the current setup.

#### Classifying custom data
You can change the input to any UTexture2D you can access in your editor or game, but if the example is using a *ConnectedTFMnistActor* you can also use your mouse/fingers to draw shapes to classify. Simply go to http://qnova.io/e/mnist on your phone or browser after your training is complete,  then draw shapes in your browser and it will send those drawn shapes to your UE4 editor for classification. 

![custom classification](http://i.imgur.com/TAV4Rie.gif)

Note that only the latest connected UE4 editor will receive these drawings, if it's not working just restart your play in editor to become the latest editor that connected. You can also host your own server with the node.js server module found under: https://github.com/getnamo/tensorflow-ue4-examples/tree/master/ServerExamples. If you want to connect to your own server, change the *ConnectedTFMnistActor->SocketIOClient->Address and Port* variable to e.g. localhost:3000.

#### Other classifiers e.g. CNN Keras model
If you want to try other mnist classifiers models, change your *ConnectedTFMnistActor->Python TFModule* variable to that python script class. E.g. if you want to try the Keras Convolutional Neural Network Classifier change the module name to [*mnistKerasCNN*](https://github.com/getnamo/tensorflow-ue4-examples/blob/master/Content/Scripts/mnistKerasCNN.py) and hit play. Note that this classifier may take around 18 min to train on a CPU, or around 45 seconds on a GPU. It should however be much more accurate than the basic softmax classifier used by default.

See available classifier models provided here: https://github.com/getnamo/tensorflow-ue4-examples/tree/master/Content/Scripts

#### Saving / Loading Models
[*mnistSaveLoad*](https://github.com/getnamo/tensorflow-ue4-examples/blob/master/Content/Scripts/mnistSaveLoad.py) python script will train on the first run and then save the trained model. Each subsequent run will then use that trained model, skipping training. You can also copy and paste this saved model to a new project and then when used in a compatible script, it will also skip the training. Use this as a guide to link your own pre-trained network for your own use cases.

You can force retraining by either changing *ConnectedTFMnistActor->ForceRetrain* to true or deleting the model found under *Content/Scripts/model/mnistSimple*

### Basic Tensorflow Example - Addition & Subtraction of Float Arrays

Map is found under [_Content/ExampleAssets/Maps/Basic.umap_](https://github.com/getnamo/tensorflow-ue4-examples/blob/master/Content/ExampleAssets/Maps/Basic.umap) 

![basic example](http://i.imgur.com/I50IQ8h.png)

Uses *TFAddExampleActor* to encapsulate [*addExample.py*](https://github.com/getnamo/tensorflow-ue4-examples/blob/master/Content/Scripts/addExample.py). This is a bare bones basic example to use tensorflow to add or subtract float array data. Press 'F' to send current custom struct data, press 'G' to change operation via custom function call. Change your default *ExampleStruct* *a* and *b* arrays to change the input sent to the tensorflow python script.

### Other Examples
If you have other examples you want to implement, consider contributing or post an [issue](https://github.com/getnamo/tensorflow-ue4-examples/issues) with a suggestion.

## Tensorflow API

See https://github.com/getnamo/tensorflow-ue4 for latest documentation.

## Dependencies

depends on: 

https://github.com/getnamo/tensorflow-ue4

https://github.com/getnamo/UnrealEnginePython 

https://github.com/getnamo/socketio-client-ue4

## Troubleshooting

### Startup Error

If you're seeing something like

![no plugins error](https://i.imgur.com/11hIUu6.png)

You did not follow [step 3. in setup](https://github.com/getnamo/tensorflow-ue4-examples#setup). Each [release](https://github.com/getnamo/tensorflow-ue4-examples/releases) has a matching plugin that you need to download and drag into the project folder.


### Video
There's a video made by github user _Berranzan_ that walks through setting up the tensorflow examples for 4.18 with GPU support.

[![Neural networks on UE4](http://img.youtube.com/vi/ZciLnYV4jIo/0.jpg)](https://www.youtube.com/watch?v=ZciLnYV4jIo)

For issues not covered in the readme see:

https://github.com/getnamo/tensorflow-ue4-examples/issues

and

https://github.com/getnamo/tensorflow-ue4/issues


## Presentation
Example project used in the presentation https://drive.google.com/open?id=1GiHmYJeZI6BKUKYfel6xc0YFhMbjSOoCY17nl98dihA contained in https://github.com/getnamo/tensorflow-ue4-examples/tree/presentation branch

Second presentation with 0.4 api: https://docs.google.com/presentation/d/1p5p6CjYYYfbflFpvr104U1GwrfArfl4Hkcb6N3SIBH8
