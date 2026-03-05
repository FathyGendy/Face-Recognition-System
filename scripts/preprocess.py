import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=14, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def create_embeddings(data_path):
    dataset = datasets.ImageFolder(data_path)
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])
    
    temp_embeddings = {name: [] for name in idx_to_class.values()}
    print("Starting Face Detection...")

    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob > 0.92:
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
            temp_embeddings[idx_to_class[idx]].append(emb)
    
    final_embeddings = []
    final_names = []

    for name, embs in temp_embeddings.items():
        if len(embs) > 0:
            mean_emb = torch.cat(embs).mean(dim=0, keepdim=True)
            final_embeddings.append(mean_emb)
            final_names.append(name)
            print(f"Created Master Profile for: {name} (used {len(embs)} images)")

    torch.save([final_embeddings, final_names], 'processed/data.pt')
    print("\nOptimized Database saved!")

if __name__ == "__main__":
    create_embeddings('data/known_faces')