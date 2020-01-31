# Real-Time Object Detection using Tensorflow Object_detection API (windows)


### This is a beginner friendly tutorial.
<img style = "margin-left: 0px;" src='data/Annotation%202020-01-23%20004710.png' width=38>

_Author: Richard Joy_  
richardjoy530@gmail.com

---
Don't get overwhelmed by the title. Seriously, anyone can do object detection using TensorFlow even with very little or no knowledge in python by following this tutorial. Tensorflow makes it much easier to do machine learning. All we need to know is how to set up all the packages and various installations, and how to not mess up with different versions of the same package or library.

Just keep on reading you will get the hang of it  :)

_NOTE: some files in this are a bit large you will be required to download around 800 Mb throughout this process if your internet connection is slow it will take a lot of time_

## 1. The setup

#### Installing Anaconda Environment

There are mainly two different variations on TensorFlow: `tensorflow`-it runs on the systems CPU and `tensorflow_gpu`- it runs on GPU, this is fast but not all GPUs are supported by TensorFlow and its installation is not quite simple.

There are various versions of TensorFlow, and this is the most annoying part that I've seen. Some python modules don't work with `tensorflow_v2.0` and others work only with `tensorflow2.0`. Therefore, I highly recommend using virtual environments for installing multiple versions of TensorFlow. Don't worry, I'll tell you how. For the virtual environment, I am using Anaconda(python3.7). There are lots of different virtual environment software out there, but here we will be using Anaconda.

Download and install Anaconda from this link: https://www.anaconda.com/distribution/#download-section keep the installation settings as it is.

#### Setting up workspace

Good practice in doing a project is keeping everything organized. Especially if you are doing this type of project. For this, let's make a folder `Tensorflow` in our `Documents` folder. We will be saving all our required resources and scripts to this folder. The paths of these folders will be used later in our programs.


#### Creating Virtual Environment

Now that you have anaconda installed, let's make an environment for our program. We will be using `tensorflow 2.0` for this tutorial. Open Anaconda Prompt from Start menu

![](data/Annotation%202020-01-22%20124206.png)

   _Throughout this tutorial, there are two types of codes. The ones that start with '!' and the ones that don't._
    
   _'!' These are shell commands. These are to be run in our Anaconda Prompt with the "Tensorflow" folder as our command line location. You can change the location of command line by using "cd Documents\Tensorflow" in the Anaconda prompt as shown here: (always exclude the '!' while executing in a shell)_
  


```python
!cd Documents\Tensorflow 
```

 ![](data/Annotation%202020-01-22%20130038.png)
    
   _The other commands are python commands. We'll see how and where to run it later in this tutorial._
    
Now create an environment by typing this in our Anaconda prompt with our folder `Tensorflow` as our command line location.


```python
!conda create -n tensorflow2 pip python=3.6 -y
```

This will create an environment named `tensorflow2` and with python3.6 as the interpreter.

Next, activate the environment by:

```python
!conda activate tensorflow2
```

Once you do this you'll see `(tensorflow2)` at the beginning of the command line, it means you are in that environment. We can create as many environments as we need, the modules that we install in an environment is exclusive to that environment. As a result, there will not be any clash between different versions of any module. Well, now we understand why it is crucial to use virtual environments. 

![](data/Annotation%202020-01-22%20132153.png)


#### Installing modules

Now that we are in the virtual env' let's install TensorFlow.

```python
!pip install tensorflow==2.*
```

we can install python libraries using `pip` or `conda`. These are some python libraries that we need in our code, so let's install it using `pip` or if it doesn't work use `conda` as shown below.

```python
!pip install pillow lxml jupyter matplotlib cython numpy 
```

if any of the above libraries did not successfully  install, then use `conda` just replace `pip` as the code:

```python
!conda install opencv -y
```

or

```python
!pip install opencv-python
```


#### Installing Protobuffers 

In simple words: Protobuffers are some formats/structures in which we can save data (like .XML and .json files) and are language independent(these files can be used by different languages). We need to compile these before we can use it.

There are various methods to install this. The safest method that I've found is shown below. Download `(eg: protoc-3.11.2-win64.zip)` and extract the latest version of protoc for windows from https://github.com/protocolbuffers/protobuf/releases

Now go to the folder `Program Files` in your `C:` drive and create a folder named `Google Protobuf` copy the extracted folders into this folder. Now it should look like this:

![](data/Annotation%202020-01-22%20100017.png)

Now we have to add this to the system path. Search `environment variables` in the `Start menu`

![](data/Annotation%202020-01-22%20095759.png)

Select `Environment Variables`

![](data/Annotation%202020-01-22%20100254.png) 

In `System variables` select `path` and edit. 

![](data/Annotation%202020-01-22%20100455.png)

Now `Add` a new path to the list `C:\Program Files\Google Protobuf\bin`

![](data/Annotation%202020-01-22%20100731.png)

_Protobuf installation is complete_


#### Downloading Tensorflow/Models

_NOTE: This download is around 500 Mb_

You can either download this folder from https://github.com/tensorflow/models/archive/master.zip and extract this zip file into our `Tensorflow` folder and rename the `model-master` folder to just `model`. The directory must be as shown below:

![](data/Annotation%202020-01-22%20200255.png)

##### or

Install `git` and clone the directory as it is. (I recommend using `git`, it's more convenient other than going to the website, downloading and then extracting..... aahhh that's a mess. )

```python
!conda install -c anaconda git -y
```

We can get the same by this shell command (Remember to run this shell commands in our `Tensorflow` folder location as we've seen before)

```python
!git clone https://github.com/tensorflow/models.git
```

#### Protobuf Compiling.

Let's compile those protobuf files that we've seen above. The `protos` files in our `Tensorflow` folder are located at `models/research/object_detection/protos/` it can be combined using a shell command. You need to open a new Anaconda shell and activate our environment by `!conda activate tensorflow2`

But unlike other commands, we should run this in prompt with location model/research

_just run these codes it should get you there._

```python
!cd Documents/Tensorflow/models/research
!protoc object_detection/protos/*.proto --python_out=.
```

if you get an error in changing the directory try: `!cd models/research` instead of the first command.


#### Building and Installing Object Detection files

Building and Installing this means: Telling our python interpreter where to search for, if a module is requested by our code. In other words, in our python code, we are importing various files from the `model` folder. So by building and installing, our python interpreter will know where to look at if such a file is requested.

_These commands must be run at `Tensorflow\models\research`. Just run these commands since we are already in this location

```python
!python setup.py build
!python setup.py install
```


### Congratulations!!! our Setup part is complete. Now, have a glass of water!!!



## 2. Python Code

You can just copy the entire code blocks below to a notepad and save it as `run.py` in our `Tensorflow` Folder. After that you can read the explanation of each code block down below:

_Copy all the codes to a text document in notepad as shown:_

![](data/Annotation%202020-01-22%20211555.png)

_Save the file as 'run.py' as shown:_

![](data/Annotation%202020-01-22%20211703.png)

_Now your 'Tensorflow' folder should look like this_

![](data/Annotation%202020-01-22%20211733.png)

_To run the file `run.py`:_

Enter the shell command given below in the anaconda prompt at the location of our 'Tensorflow' folder. This will take some time just wait. Press 'q' to terminate the program

```python
!cd..
!cd..
!python run.py
```


#### 1. Importing required modules for our program 

There is nothing much to say in here, it's just telling the interpreter that these are the files that we need to run this program.

_if you come across any error here, then that means that module is not installed in the environment. You can fix it by running ' !pip install _______ ' in the prompt._

```python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
```

Next, we are importing the modules that we build and installed before. If u get an error here, its because the building and installing was not successful

```python
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
```


#### 2. Mapping old functions to new

There are still some modules that use `tensorflow_v1` and we have installed `tensorflow_v2` so these bits of code just maps the old functions the corresponding functions in the new version.

```python
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile
```

#### 3. Defining a function to Load a given model

Let's have a quick overview of what exactly is a model. It is the brain of the object detection program. Basically, it contains a neural network which helps in detecting the object in a given image (actually it's called a tensor, we'll see it later). There are a wide variety of models out there. Some are fast, and some are lightweight(means it does not need much computing power, it can be used in smartphones). For this tutorial, we will be using models that already exist and that are tested by experts.

`load_model` is a generic function that takes a parameter `model_name` which is the name of the pre-trained models.
`model_name` can be the name of any of the trained models in this link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

This function downloads the model from the internet and  saves it in our `Tensorflow` folder

```python
def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model
```

`PATH_TO_LABELS` is the location of the `mscoco_lable_map.pbtxt`. It is a file that has all the names and tags of the detectable objects. ie, if the program detects an object, it looks in the label map to find its name.

`category_index` is like a dictionary which stores all the name of the objects in the `mscoco_lable_map.pbtxt` and its tag(a number)

```python
# List of the strings that are used to add the correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
```

#### 4. Loading the model

This is where we run the function `load_model` that we made above. Notice the `model_name` here we use the model: `ssd_inception_v2_coco_2017_11_17` or use `ssd_mobilenet_v1_coco_2018_01_28` if the other one is slow. You can see these names in the link above. And this model is saved in `detection_model`

```python
model_name = 'ssd_inception_v2_coco_2017_11_17'
detection_model = load_model(model_name)
```

#### 5. Defining a function to detect objects

`run_inference_for_single_image` is a function that takes a `model` and an `image`. It uses the given `model` to detect the objects in the given `image`

```python
def run_inference_for_single_image(model, image):
    
    # The image is converted to a number array for computation and this array replaces the image in the
    # variable 'image'.
    image = np.asarray(image)
    
    # The number array is converted to a tensor object for tensorflow to work, this tensor object is our input
    # to the model and is saved in 'input_tensor'
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    
    # This is where the MAGIC happens. Our image processed in the form of a tensor object is given to the
    # model. 
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                   for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict
```

#### 6. Detecting frame by frame

`cv2` is a module that is required to make interactions with our camera and display images onto our screen. `cap` is an object of cv2 from which we can get the images frame by frame. 


This set of code is just taking each frame of the video image by image and passing it to the `run_inference_for_single_image` function that we defined above. This process is looped in the `while` loop and this is the reason for the continuous video detection. Our model detects objects in a single image, so we repeat this process for each frame and display it on the screen with the function `cv2.imshow`. Further explanation is given within the code.


```python
import cv2
# 'VideoCapture()' is used for getting the video input from a device(in this case, our webcam ) 
#  or a video file (place the file name in the brackets to do so.) and this is saved in a variable 'cap'
cap = cv2.VideoCapture(0)

# 'try' is a way of telling the interpreter just so it dosn't freak out and throw errors 
#  when it encounters a problem.
try:
    def run_inference(model, cap):
        while True:
            # This is where  we capture each frame image by image in a variable called 'image_np'.
            ret, image_np = cap.read()
            # this image along with the model is given to the previous function as it gives the detection
            # details the output is stored in 'output_dict' and it contains the objects name, 
            # probablity score, and the boundary box
            output_dict = run_inference_for_single_image(model, image_np)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
                
            # 'cv2.imshow' just displays the image with detected boundaries
            cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
            
            #This code just stops the program if we press 'q' on our keyboard
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


except Exception as e:
    cap.release()
run_inference(detection_model, cap)
```

#### Congratulations you've made it to the end.

#### If you have got the final output, just take a moment and appreciate yourself for doing this

#### If you did not get the output, find out what went wrong, take it as a challenge. This way you will learn a lot.

---

# The End

I will be making similar tutorials on Training object detection on custom objects. You can always find my tutorials at my GitHub-repo: https://github.com/richardjoy530/object_detection_tutorials

I will be updating this file and you can get it from https://github.com/richardjoy530/object_detection_tutorials/blob/master/realtime_object_detection_windows.md

Feel free to contact me if you ran into some troubles or need further info: richardjoy530@gmail.com

---

code refrences taken from :

    _https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
     https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api
     https://github.com/EdjeElectronic_
