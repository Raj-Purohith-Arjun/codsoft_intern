import torch
from PIL import Image

import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.nn.utils.rnn import pack_padded_sequence
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.nn import functional as F
from torch.optim import Adam
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.nn.utils.rnn import pack_padded_sequence
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.nn import functional as F
from torch.optim import Adam
from PIL import Image  # Add this import statement

# Set up NLTK for tokenization
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rest of your code follows...
captions = []

with open(r'C:\Users\91967\Documents\codsoft\captions.txt', "r") as file:
    for line in file:
        captions.append(line.strip())



# Set up NLTK for tokenization
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        return features

# Example transformation to preprocess images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load an image and apply the transformation
image = transform(Image.open('C:\\Users\\91967\\Documents\\codsoft\\example.jpg').convert("RGB")).unsqueeze(0).to(device)

# Create the image encoder
image_encoder = ImageEncoder().to(device)
image_features = image_encoder(image)

# Load and preprocess caption data
captions = ...  # Your caption data
captions = [caption.lower() for caption in captions]

# Tokenize the captions
all_words = [word for caption in captions for word in word_tokenize(caption)]
word_counter = Counter(all_words)

# Create a vocabulary mapping
vocab = ["<unk>", "<pad>", "<start>", "<end>"] + [word for word, count in word_counter.items() if count >= 5]

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, image_features, captions, word_to_idx, max_caption_length=20):
        self.image_features = image_features
        self.captions = captions
        self.word_to_idx = word_to_idx
        self.max_caption_length = max_caption_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        caption = word_tokenize(caption)
        caption = ["<start>"] + caption + ["<end>"]

        caption_idx = [self.word_to_idx.get(word, self.word_to_idx["<unk>"]) for word in caption]

        padding_length = self.max_caption_length - len(caption_idx)
        caption_idx += [self.word_to_idx["<pad>"]] * padding_length

        return {
            "image_features": self.image_features[idx],
            "caption": torch.tensor(caption_idx)
        }

# Create the dataset and data loader
dataset = CaptionDataset(image_features, captions, word_to_idx)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CaptionGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.rnn(embeddings)
        outputs = self.fc(lstm_out)
        return outputs

# Create the caption generator
embedding_dim = 256
hidden_dim = 512
caption_generator = CaptionGenerator(vocab_size, embedding_dim, hidden_dim).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = Adam(caption_generator.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for batch in data_loader:
        image_features_batch = batch["image_features"].to(device)
        captions_batch = batch["caption"].to(device)

        optimizer.zero_grad()
        outputs = caption_generator(image_features_batch, captions_batch[:, :-1])
        targets = captions_batch[:, 1:].contiguous().view(-1)
        loss = criterion(outputs.view(-1, vocab_size), targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def generate_caption(image_features, caption_generator, max_length=20):
    caption = ["<start>"]
    for _ in range(max_length):
        caption_input = torch.tensor([word_to_idx[caption[-1]]]).unsqueeze(0).to(device)
        output = caption_generator(image_features, caption_input)
        predicted_idx = output.argmax(2)[:, -1].item()
        predicted_word = idx_to_word[predicted_idx]
        caption.append(predicted_word)
        if predicted_word == "<end>":
            break
    return " ".join(caption[1:-1])  # Exclude <start> and <end>

generated_caption = generate_caption(image_features, caption_generator)
print("Generated Caption:", generated_caption)

