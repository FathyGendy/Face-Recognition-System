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
    print("Error: Run preprocess.py first!")
    exit()

embedding_list, name_list = torch.load('processed/data.pt')

def identify_face(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Could not read image path")
        return
    
    img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is not None:
        faces = mtcnn(img_rgb)
        embeddings = resnet(faces.to(device)).detach().cpu()

        for i, emb in enumerate(embeddings):
            dist_list = [torch.dist(emb, known_emb).item() for known_emb in embedding_list]
            
            min_dist = min(dist_list)
            min_dist_idx = dist_list.index(min_dist)
            name = name_list[min_dist_idx]

            if min_dist > 0.90:
                name = "Unknown"

            box = boxes[i].astype(int)
            cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{name} ({min_dist:.2f})", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"Detected: {name} with distance {min_dist:.2f}")

        cv2.imshow("Identification Result", img_bgr)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    else:
        print("No faces detected in this image.")

if __name__ == "__main__":
    img_to_test = input("Enter the path to the image (e.g., test.jpg): ")
    identify_face(img_to_test)