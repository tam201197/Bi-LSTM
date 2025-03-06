import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

dev_path = "./data/dev.conll"
test_path = "./data/test.conll"
train_path = "./data/train.conll"


def preprocessing_data(data_path, multi_sentences=False):
    dataset = {
        "doc": [],
        "labels": []
    }
    with open(data_path, 'r') as file:
        lines = file.readlines()
    texts = []
    labels = []
    for line in lines:
        if (line == "\n"):
            continue
        line = line.replace("\n",'')
        if line.startswith('-DOCSTART-'):
            if texts:
                dataset["doc"].append(texts)
                dataset["labels"].append(labels)
            texts = []
            labels = []
        else:
            tokens = line.split("\t")
            if multi_sentences: 
                texts.append(tokens[0])
                labels.append(tokens[-1])
            else: 
                if tokens[0] == '.':
                    if texts:
                        dataset["doc"].append(texts)
                        dataset["labels"].append(labels)
                    texts = []
                    labels = []
                else:
                    texts.append(tokens[0])
                    labels.append(tokens[-1])
            
    #new_df = df.drop(columns=['-X-', '-X-'])
    #new_df.columns = ['text', 'label']
    return dataset

def read_embeddings(embedding_path):
    # extract the embedding dimension from the filename (simpler solutions are imaginable)
    embeddings = defaultdict(list)
    with open(embedding_path, "r", encoding="utf-8") as file:
          for line in file:
            line = line.replace("\t", ' ')
            line = line.strip()
            if line:
                line_elements = line.split(' ')
                embeddings[line_elements[0]] = [float(element) for element in line_elements[1:]]
    return embeddings

def embed_data(sentences):
    default_embedding = [0.0] * 50
    embedded_sentence = [embeddings.get(word, default_embedding) for word in sentences]
    return torch.tensor(embedded_sentence)

def embed_labels(labels):
    result = []
    for value in labels:
        label_vector = [0.0] * len(tag)
        label_vector[tag2idx[value]] = 1.0
        result.append(label_vector)
    return torch.tensor(result)

class NERDataset(Dataset):
    def __init__(self, dataset, device):
        self.labels = []
        self.lengths = []
        self.data = []
        for text in tqdm(dataset["doc"]):
            embeddings = embed_data(text)
            self.data.append(embeddings.to(device))
            self.lengths.append(len(embeddings))
        for l in tqdm(dataset["labels"]):
            #embeddings = embed_labels(l)
            embeddings = torch.tensor([tag2idx[value] for value in l], dtype=torch.long)
            self.labels.append(embeddings.to(device))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embeddings = self.data[idx]
        label = self.labels[idx]
        length = self.lengths[idx]

        return embeddings, length, label

def create_dataloader(dataset, device, batch_size=1, shuffle=True):
    imdb_dataset = NERDataset(dataset, device)
    dataloader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bi_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, input_word_embedding):
        bi_lstm_output, _ = self.bi_lstm(input_word_embedding)
        return self.fc(bi_lstm_output)

def train(model, num_epochs, loss_function, optimizer,
              train_data, test_data, device):
    epoch_loss_logger = []
    epoch_f1_scores = []
    best_f1 = 0
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1} of {num_epochs}")
        # training
        train_loss = []
        model.train()
        print("\t Training progress: \n")
        for embedding, length, label in tqdm(train_data):
            optimizer.zero_grad()

            # Forward pass
            predictions = model(embedding)

            # Reshape for loss calculation
            predictions = predictions.view(-1, predictions.shape[-1])  
            label = label.view(-1)  

            # Compute loss
            loss = loss_function(predictions.to(device=device), label.to(device=device))

            #train_prediction = model(embedding)
            #loss = loss_function(train_prediction.to(device=device), label.to(device=device))
            train_loss.append(loss)
            loss.backward()
            optimizer.step()

        epoch_loss_logger.append(torch.mean(torch.tensor(train_loss)))

        
        # evaluation
        print("\n\t Evaluation progress: \n")
        with torch.no_grad():
            model.eval()
            predictions = []
            targets = []

            for embedding, length, label in tqdm(test_data):
                outputs = model(embedding)
                test_outputs = torch.argmax(outputs, dim=2)
                predictions.extend(test_outputs.view(-1).tolist())
                targets.extend(label.view(-1).tolist())


            train_f1_score = f1_score(predictions, targets, average='macro')
            print("\t Test F1 Score in epoch " + str(epoch) + ": " + str(train_f1_score)
                    + " Train loss: " + str(epoch_loss_logger[epoch].item()))
            epoch_f1_scores.append(train_f1_score)
            if train_f1_score > best_f1:
                best_f1 = train_f1_score
                torch.save(model.state_dict(), "best_model.pth")
                print(f"at epoch {epoch +1} best model saved (F1-score improved: {best_f1:.4f})")

        
    
    return epoch_loss_logger, epoch_f1_scores, model

def visualize_data(data, title, lerning_rate):
    fig_name = title.replace(' ', '_')
    x_axis = np.arange(num_epochs) + 1
    plt.plot(x_axis, data)
    plt.title(title)
    plt.savefig(f'{fig_name}_lr_{lerning_rate}.png')
    plt.clf()


hidden_size = 100
num_epochs = 20



embeddings_path = "glove.6B.50d.txt"
embeddings = read_embeddings(embeddings_path)
embeddings_size = len(embeddings[next(iter(embeddings))])
df = pd.read_csv(train_path, sep="\t", header=None, names=["text", "POS", "Chunk", "label"], quoting=3)
df_nan = df[df["label"].notna()]
tag = df_nan['label'].unique()
tag2idx = {value: index for index, value in enumerate(tag)}
output_size = len(tag)


input_size = embeddings_size 

def model_evaluate(model, eval_dataset):
    print("\n\t Evaluation progress: \n")
    with torch.no_grad():
        model.eval()
        predictions = []
        targets = []

        for embedding, length, label in tqdm(eval_dataset):
            outputs = model(embedding)
            test_dataloader = torch.argmax(outputs, dim=2)
            predictions.extend(test_dataloader.view(-1).tolist())
            targets.extend(label.view(-1).tolist())


        train_f1_score = f1_score(predictions, targets, average='macro')
    return train_f1_score

def run(lr, multi_sentences=False):
    dev_dataset = preprocessing_data(dev_path, multi_sentences)
    train_dataset = preprocessing_data(train_path, multi_sentences)
    test_dataset = preprocessing_data(test_path, multi_sentences)
    print("Create dataloader for training")
    train_dataloader = create_dataloader(train_dataset, device, 1)

    print("Create dataloader for testing")
    test_dataloader = create_dataloader(test_dataset, device, 1)

    print("Create dataloader for developing")
    dev_dataloader = create_dataloader(dev_dataset, device, 1)

    bi_lstm = BiLSTM(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bi_lstm.parameters(), lr=lr)
    loss_function = criterion.to(device)
    train_losses, f1_scores, model = train(bi_lstm, num_epochs, loss_function, optimizer, train_dataloader, dev_dataloader, device)
    visualize_data(train_losses, "Epoch losses", lr)
    visualize_data(f1_scores, "F1 scores",lr)
    
    print(f"Final macro-averaged F1 after 20 epochs: {model_evaluate(model, test_dataloader)}")

    
    model = BiLSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("best_model.pth"))  # Load best model
    model.to(device)
    print(f"Final macro-averaged F1 with early stoping: {model_evaluate(model, test_dataloader)}")

run(0.0007, multi_sentences=True)

#Bi-LSTM with single sentence

    # macro-averaged F1 scores after trained for 20 epoches with lr 0.0007 : 0.6042290590761805
    # macro-averaged F1 scores using early stopping with lr 0.0007 : 0.6242508059608585

    # macro-averaged F1 scores after trained for 20 epoches with lr 0.0001 : 0.6165566758295815
    # macro-averaged F1 scores using early stopping with lr 0.0001 : 0.6165566758295815

    # macro-averaged F1 scores after trained for 20 epoches with lr 0.0005 : 0.6132677129993109
    # macro-averaged F1 scores using early stopping with lr 0.0005 : 0.6300586305443221

    # macro-averaged F1 scores after trained for 20 epoches with lr 0.0003 : 0.6134956384803433
    # macro-averaged F1 scores using early stopping with lr 0.0003 : 0.6437442288268493

#Bi-LSTM with multiple sentences

    # macro-averaged F1 scores after trained for 20 epoches with lr 0.0003 : 0.6160199819234193
    # macro-averaged F1 scores using early stopping with lr 0.0003 : 0.612160217512418

    # macro-averaged F1 scores after trained for 20 epoches with lr 0.0007 : 0.6292275905489979
    # macro-averaged F1 scores using early stopping with lr 0.0007 : 0.6297390615811478