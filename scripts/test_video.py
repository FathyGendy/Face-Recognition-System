import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

if not os.path.exists('processed/data.pt'):
    print("Error: data.pt not found. Run preprocess.py first!")
    exit()

embedding_list, name_list = torch.load('processed/data.pt')

def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    print(f"Processing video: {video_path}")
    print("Press 'q' to stop the video playback.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video reached.")
            break
        
        img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            faces = mtcnn(img_rgb)
            
            if faces is not None:
                embeddings = resnet(faces.to(device)).detach().cpu()
                
                for i, emb in enumerate(embeddings):
                    dist_list = [torch.dist(emb, known_emb).item() for known_emb in embedding_list]
                    
                    min_dist = min(dist_list)
                    min_dist_idx = dist_list.index(min_dist)
                    name = name_list[min_dist_idx]
                    
                    if min_dist > 0.90:
                        name = "Unknown"
                    
                    box = boxes[i].astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    label = f"{name} ({min_dist:.2f})"
                    cv2.putText(frame, label, (box[0], box[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Video Identification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = input("Enter the path to the video file (e.g., my_video.mp4): ")
    if os.path.exists(path):
        process_video(path)
    else:
        print("File path does not exist!")