# Navarasa-Detection

The Navarasa System is an AI-powered emotion recognition model that detects and classifies human facial expressions into the nine classical Indian Navarasa emotions in real time.
The system processes real-time video streams, detects human faces, analyzes facial features, and classifies the expressed emotion into one of the nine Navarasas: 
Shringara, Hasya, Karuna, Raudra, Veera, Bhayanaka, Bibhatsa, Adbhuta, and Shanta.

This project utilizes the webcam of our laptops to recognize the emotion giving a 3 second timestamp for the person. The system detects the emotion within this time, and lock on this emotion unless it is reset. 
The model automatically detects the face in each frame and ignores background and non-face regions. Then, crops the selected face region, resize it and normalize for deep learning compatibility. The system uses a CNN-based deep learning model to extract the facial features such as eye movement, mouth shape, eyebrow tension etc.. Then, classifies the face into one of the 9 Navarasa categories. 
