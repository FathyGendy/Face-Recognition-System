import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

if not torch.os.path.exists('processed/data.pt'):
    print("Error: data.pt not found. Run preprocess.py first!")
    exit()

load_data = torch.load('processed/data.pt')
embedding_list = load_data[0] 
name_list = load_data[1] 

video = cv2.VideoCapture(0)
print("Starting Webcam... Press 'q' to exit.")

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            face = mtcnn(img)
            
            if face is not None:
                embeddings = resnet(face.to(device)).detach().cpu()
                
                for i, emb in enumerate(embeddings):
                    dist_list = []
                    
                    for known_emb in embedding_list:
                        dist = torch.dist(emb, known_emb).item()
                        dist_list.append(dist)
                    
                    min_dist = min(dist_list)
                    min_dist_idx = dist_list.index(min_dist)
                    name = name_list[min_dist_idx]
                    
                    if min_dist > 0.90:
                        name = "Unknown"
                    
                    curr_box = boxes[i].astype(int)
                    frame = cv2.rectangle(frame, (curr_box[0], curr_box[1]), (curr_box[2], curr_box[3]), (0, 255, 0), 2)
                    
                    label = f"{name} ({min_dist:.2f})"
                    frame = cv2.putText(frame, label, (curr_box[0], curr_box[1] - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Surveillance System - Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()