1. What is the computer vision ?
2. What is the goal of computer vision ?

> To  **extract meaning from the picture, meaning from pixels** .

3. What are the main challenges to achieve the goal of computer vision ?

> While humans see a scene (e.g., a train crash), a computer sees only a **matrix of numbers of pixels. So, The core goal is to enable the computer to perceive the world as humans do**

4. What are the different types of information needed to extract out of images ?

> Given an image or video, Computer Vision tasks generally aim to extract two kinds of information
>
>> 1. **Semantic Information**: Related to understanding objects and context, similar to grammar (e.g., recognizing what an object is).
>> 2. **Metric Information**: Related to measurement (e.g., distance, 3D structure).
>>

5. What are the semantic computer vision tasks ?

> These are the tasks focus on understanding the content, context and objects within an image.
>
>> * **Labelling and Localization** : Identifying objects (e.g., buses, cars, flags) and localizing their position in the image.
>> * **Scene Categorization** : Analyzing the overall scene context (e.g., identifying a picture as an outdoor city scene with traffic).
>>   **Object Recognition** A simple task where the image has  **one dominant object** , and the goal is to label its class (e.g., a container ship or a leopard)
>>
>>> * *Example:* The ImageNet challenge
>>>
>>
>> * **Object Detection** More challenging than recognition, as it involves localizing (drawing a bounding box) and naming **multiple objects** in a scene
>>
>>> * *Practical Examples:* Face detection in cameras detecting pedestrians and cars in traffic and Number Plate Readers (localizing the plate and recognizing the letters)
>>>
>>
>> * **Image Segmentation** The process of labeling or segmenting **each pixel** to belong to a semantically similar object/class (e.g., clustering all pixels belonging to the road together, all pixels of a car together)
>>
>>> * *Use Case:* Essential for automatic driving to identify the drivable road, pedestrians, and other vehicles
>>>
>>
>> * **Other Tasks:**
>>
>>> * Object Tracking (following objects in a video)
>>> * Pose Estimation (estimating the pose of a person)
>>> * Activity Recognition (identifying what activity is being performed)
>>> * Image Captioning An image-to-language translation where the content of an image is summarized using natural language (e.g., "A person riding on a motorcycle on a dirt road")
>>>
>>

6. What are the metric computer vision tasks ?

> These are the tasks that uses vision as measurement device, often for the spatial information
>
>> * **Measuring Distance/Depth** Finding the distance of objects.
>>
>>> * **Stereo Vision** Using two cameras to extract depth, similar to how human eyes work (our eyes are stereo)
>>> * **Depth from a Single Image** Getting depth perception from one image by utilizing cues like occlusion, size, and shadow
>>>
>>
>> * **Structure for Motion (SFM)** Using multi-view images or a video to find the **3D structure** of a scene and the **motion of the camera**
>> * **3D Model Creation** Creating a 3D model of a real-world object (like the Taj Mahal) from a collection of images.
>>

7. What are the geometric computer vision tasks ?

> These techniques are used for 3D reconstruction and measurement.
>
>> **Depth from Stereo** Using two images to find point correspondence and then using triangulation to accurately measure the distance (depth) of an object from the camera.
>> **Depth from a single image**
>> **Structure for Motion (SFM)** Taking multiple images from various viewpoints (or a video) to reconstruct the entire 3D model of the scene and estimate the camera's location and pose
>> **Photometric Stereo** The camera is kept fixed while the **lighting is changed** to capture multiple images. The variation in shading cues is used to find the  **surface normal** , which in turn helps in reconstructing the surface depth.
>>

8. What are the main challenges in Computer Vision ?

> Computer vision algorithms must be robust to the following factors
>
> | **Challenge**              | **Description**                                                                                         |
> | -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
> | **Viewpoint Variation**    | The same object is seen from different orientations (e.g., a tilted face vs. an upright one).                 |
> | **Illumination Variation** | The same scene captured under different lighting conditions, causing drastic pixel variations (e.g., shadows) |
> | **Scale Variation**        | The same object appears at different sizes in an image.                                                       |
> | **Deformation**            | The object's shape can change (e.g., a person or animal changing its pose while running)                      |
> | **Occlusion**              | Part of the object is hidden by another object in the foreground.                                             |
> | **Clutter**                | A busy background (background clutter) makes it difficult to detect the target object.                        |
> | **Motion Blur**            | Blurring of the image caused by moving objects.                                                               |
> | **Intra-class Variation**  | Objects within the same class (e.g., a chair) come in a huge variety of shapes, colors, and designs           |

9. Can you tell me about Standard Computer Vision Tasks ?

> 1. Object Recognition : ImageNet Challenge
> 2. Object Detection(Recognition + Localization) : YOLO(You Only Look Once) Real-time object detection
> 3. Face detection + Localization + Classification : Security Camera
> 4. Optical Character Recognition(OCR) : Number plate reader
> 5. Image Segmentation : Segment each pixel that are similar semantically.
> 6. Object Tracking :
> 7. Pose Estimation :
> 8. Activity Recognition :
> 9. Image Captioning : What is the content in the image and label it with the text

10. How did computer recognize an image ?
