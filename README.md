<h1 align="center">Object Detection and Text Recognition App</h1>

<h2>This Android application leverages TensorFlow Lite and ML Kit to perform real-time object detection and text recognition using the device's camera. It supports both CPU, GPU, and NNAPI delegates for improved performance on compatible hardware.</h2>


<h3>Features</h3>


<h4>Real-time Object Detection: Uses TensorFlow Lite models (EfficientDet and MobileNet) to detect and label objects in the camera feed.</h4>
<h4>Text Recognition: Utilizes Google's ML Kit to recognize and read text from the camera feed.</h4>
<h4>Voice Commands: Controls object detection and text recognition modes using speech commands.</h4>
<h4>Configuration Options: Allows adjustment of detection thresholds, maximum results, and threading options through UI controls.</h4>


<h3>Components</h3>



<h4>ObjectDetectorHelper: Manages the configuration and execution of the TensorFlow Lite object detection model.</h4>
<h4>CameraFragment: The main fragment handling camera setup, image analysis, and integrating object detection and text recognition functionalities.</h4>
<h4>Speech Recognition: Listens for specific voice commands to start/stop object detection and text recognition.</h4>
<h4>Text-to-Speech: Reads out detected text and object labels using Android's Text-to-Speech engine.</h4>


<h3>Usage</h3>
<h4>Start the app and grant necessary permissions (Camera and Microphone).</h4>
<h4>Use vocal commands like "start object detection" or "read text" to begin detection/recognition.</h4>
<h4>Control detection/recognition parameters through the provided UI controls.</h4>
