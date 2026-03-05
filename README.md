# Face Recognition Surveillance System

A real-time Computer Vision surveillance system built using **PyTorch** and **Deep Learning**. This system is designed to detect, align, and identify individuals from a live webcam feed, static images, or video files. It features an automated "Master Profile" generation system to ensure high accuracy and handles "Unknown" individuals using distance thresholding.

---

## Features
*   **Real-Time Identification:** Process live video streams via webcam.
*   **Face Detection & Alignment:** Uses **MTCNN** to detect faces and align them using 5 facial landmarks (eyes, nose, mouth) for better extraction.
*   **Feature Extraction:** Utilizes a pre-trained **InceptionResnetV1** (trained on VGGFace2) to generate 512-dimensional face embeddings.
*   **Master Profile Logic:** Automatically calculates the **Mean Embedding** of multiple photos (10-15 per person) to create a robust identity profile.
*   **Unknown Detection:** Implements a Euclidean distance threshold to classify unidentified individuals as "Unknown."
*   **Static Forensic Analysis:** Dedicated scripts for processing standalone images and video files.

---

## Tech Stack
*   **Framework:** PyTorch
*   **Libraries:** `facenet-pytorch`, `OpenCV`, `NumPy`, `Pillow`
*   **Language:** Python 3.11+
*   **Models:** MTCNN (Detection), InceptionResnetV1 (Recognition)

---

## 📂 Project Structure
```text
Face Recognition/
├── data/
│   └── known_faces/      # Store folders of people here (e.g., /Fathy)
├── scripts/
│   ├── preprocess.py     # Detects faces and creates the Master Profiles
│   ├── realtime.py       # Live webcam surveillance
│   ├── test_image.py     # Static image identification
│   └── test_video.py     # Video file analysis
├── processed/
│   └── data.pt           # Saved face embeddings (The "Database")
└── requirements.txt      # Project dependencies
```
## ⚙️ Installation & Setup
* Clone the Repository:
  ```bash
  git clone https://github.com/FathyGendy/Face-Recognition-System.git
  cd "Face Recognition"
  
* Create a Virtual Environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Mac/Linux
  .\venv\Scripts\activate   # Windows

* Install Dependencies:
  ```bash
  pip install torch torchvision facenet-pytorch opencv-python pandas requests tqdm scipy matplotlib

## How to Use
* Step 1: Prepare the Data - Add folders for each person you want to recognize in data/known_faces/. Aim for 10-15 images per person from different angles. Example: data/known_faces/Fathy/img1.jpg
* Step 2: Build the Database - Run the preprocessing script to find faces, align them, and generate the Master Profiles (embeddings).
  ```bash
  python scripts/preprocess.py
* Step 3: Run Surveillance
  * **Live Webcam:** python scripts/realtime.py
  * **Test Image:**  python scripts/test_image.py
  * **Test Video:**  python scripts/test_video.py
## Logic & Pipeline
* **Capture:** System grabs a frame from the input source.
* **Detection:** MTCNN scans for faces and returns bounding boxes.
* **Alignment:** Faces are rotated and cropped based on facial landmarks.
* **Embedding:** The cropped face is converted into a 512-bit vector.
* **Matching:** The live vector is compared against data.pt using Euclidean Distance.
* **Classification:**
     * Distance < 0.90 ➡️ Recognized Name.
     * Distance > 0.90 ➡️ "Unknown".
 
---

## Accuracy & Threshold Logic
In this project, I utilize **Euclidean Distance** to compare the live face embedding against the stored "Master Profiles."

### The Threshold (0.9)
Through empirical testing, a distance threshold of **0.9** was selected as the optimal "sweet spot."
*   **Why 0.9?** 
    *   **Lowering the threshold (e.g., 0.6):** Makes the system too strict. It might fail to recognize a known person if the lighting changes slightly or if they aren't looking directly at the camera (**False Negative**).
    *   **Increasing the threshold (e.g., 1.2):** Makes the system too "forgiving." It might incorrectly identify a stranger as one of the authorized users (**False Positive**).
*   **The Master Profile Advantage:** By using the **Mean Embedding** (the average of 15 photos) in `preprocess.py`, I significantly reduced "noise" in the data. This allowed me to keep the threshold at a robust 0.9 while maintaining high recognition stability.

### Factors Influencing Accuracy
1.  **Lighting:** Dim environments increase noise in the feature extraction process.
2.  **Occlusions:** Face masks or heavy scarves significantly increase the distance, often triggering an "Unknown" classification.
3.  **Alignment:** The built-in MTCNN alignment ensures that even tilted heads are normalized, keeping the distance within the target range.

## Contributors
Built with ❤️ by [Fathy](https://github.com/FathyGendy)


  

