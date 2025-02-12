from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision import transforms

# Hyperparameters
num_clients = 5
num_clusters = 2
num_rounds = 10
local_epochs = 5
batch_size = 32
learning_rate = 0.01
mu = 2
lam = 1
D = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


# Custom Dataset for HAM10000
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dirs, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dirs = img_dirs
        self.transform = transform

        # Convert diagnosis to numerical labels
        self.diagnosis_mapping = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
            'mel': 4, 'nv': 5, 'vasc': 6
        }
        self.data_frame['label'] = self.data_frame['dx'].map(
            self.diagnosis_mapping)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_id = self.data_frame.iloc[idx]['image_id'] + '.jpg'

        image_path = None
        for img_dir in self.img_dirs:
            temp_path = os.path.join(img_dir, img_id)
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            raise FileNotFoundError(
                f'Image {img_id} not found in any of the directories')

        image = Image.open(image_path).convert('RGB')
        label = self.data_frame.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label


def get_ham10000_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    img_dirs = [
        './HAM10000_images_part_1',
        './HAM10000_images_part_2'
    ]

    dataset = HAM10000Dataset(
        csv_file='HAM10000_metadata.csv', img_dirs=img_dirs, transform=transform)

    generator = torch.Generator(device=device)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=generator
    )

    return train_loader, test_loader


class ModifiedLeNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ModifiedLeNet, self).__init__()

        # Feature extractor part of the network
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Embedding layer to reduce the dimensionality
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 512),
            nn.ReLU()
        )
        # Classifier layer to output the final class probabilities
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.embedding_layer(x)
        x = self.classifier(x)
        return x

    def get_embedding(self, x):
        x = self.feature_extractor(x)
        x = self.embedding_layer(x)
        return x


# Initializing the data loaders
train_loader, test_loader = get_ham10000_loaders(batch_size)

# Initializing the clients and clusters
clients = {i: ModifiedLeNet().to(device) for i in range(num_clients)}
embedding_size = 512
clusters = {k: torch.randn(embedding_size, device=device,
                           requires_grad=True) for k in range(num_clusters)}
global_embedding = torch.zeros(embedding_size, device=device)
M = None


def compute_similarity(model_a, model_b):
    global M

    params_a = [param.data.view(-1) for name,
                param in model_a.items() if 'weight' in name]
    model_a_flat = torch.cat(params_a).to(device)

    if isinstance(model_b, dict):
        params_b = [param.data.view(-1) for name,
                    param in model_b.items() if 'weight' in name]
        model_b_flat = torch.cat(params_b).to(device)
    else:
        model_b_flat = model_b.view(-1)

    min_size = min(model_a_flat.size(0), model_b_flat.size(0))
    model_a_flat, model_b_flat = model_a_flat[:
                                              min_size], model_b_flat[:min_size]

    if M is None or M.shape[0] != min_size:
        M = torch.randn((min_size, D), device=device)

    proj_a, proj_b = model_a_flat @ M, model_b_flat @ M
    similarity = torch.dot(proj_a, proj_b) / \
        (torch.norm(proj_a) * torch.norm(proj_b) + 1e-8)
    return similarity


# Local training function for each client
def local_train(client_id, cluster_id):
    model = clients[client_id].to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model.get_embedding(imgs)
            outputs = model.classifier(embeddings)

            task_loss = criterion(outputs, labels)
            cluster_loss = mu * \
                torch.norm(embeddings.mean(0) - clusters[cluster_id])
            global_loss = lam * \
                torch.norm(embeddings.mean(0) - global_embedding)
            total_loss = task_loss + cluster_loss + global_loss

            total_loss.backward()
            optimizer.step()

    return model.state_dict()


def evaluate_model():
    for client_id in range(num_clients):
        model = clients[client_id].to(device)
        model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Client {client_id} Test Accuracy: {accuracy:.2f}%")

    print("✅ Evaluation complete.")


if __name__ == '__main__':
    print("🚀 Starting training...")
    print(f"🔥 Using device: {device}")
    print(f"ℹ️ Device name: {torch.cuda.get_device_name(0)}")
    print(
        f"Is model on GPU? {'cuda' in str(next(clients[0].parameters()).device)}")
    print(f"Is a sample tensor on GPU? {torch.randn(1).to(device).device}")
    for round in range(num_rounds):
        print(f"\n🟢 Round {round+1}/{num_rounds}")
        cluster_assignments = {i: 0 for i in range(
            num_clients)}
        for i in tqdm(range(num_clients), desc=f"Training Clients (Round {round+1})"):
            clients[i].load_state_dict(local_train(i, cluster_assignments[i]))

    print("\n🚀 Training complete!")
    evaluate_model()
