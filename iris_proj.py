import torch
import torch.nn as nn
from sklearn.datasets import load_iris
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, classification_report
import logging, os, time
import joblib, pickle


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("using {device}")

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

#now df contains the features and the target columns
#print(df.head())

#parition
X = df[data.feature_names].values #features as numpy array
y = df['target'].values  #target as numpy array
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#into torch tensors
x_train_torch = torch.tensor(x_train, dtype=torch.float32)
x_test_torch = torch.tensor(x_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

class linear_mlp(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.linear_model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,3) #3 classes for iris
        )

    def forward(self, x):
        return self.linear_model(x)
    

def quick_train(model, x_train_torch, y_train_torch, n_factors = 20, n_epochs = 20, batch_size = 512, device = "mps"):

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criteron = nn.CrossEntropyLoss()

    model.to(device)

    #create tensor dataset/dataloader from those methods
    train_dataset = TensorDataset(x_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_model_state = None 
    epoch_loss = []

    for epochs in range(n_epochs):
        model.train()
        running_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device, dtype = torch.float32)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criteron(logits, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
        
        avg_loss = running_loss/ len(train_loader.dataset)
        epoch_loss.append(avg_loss)

        best_model_state = model.state_dict() #save the best model weights
        if (epochs+1) %1 ==0:
            logging.info(f"Epoch {epochs+1}/{n_epochs} Train loss: {avg_loss:.4f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
    return model, best_model_state
    
def evaluate(model, x_test_torch, y_test_torch, batch_size):

    model.eval()
    model.to(device)
    criteron = nn.CrossEntropyLoss()

    #into dataset
    test_set = TensorDataset(x_test_torch, y_test_torch)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    all_p, all_y, all_pred = [], [], []

    with torch.no_grad():
        for xb,yb in test_loader:
            xb = xb.to(device, dtype = torch.float32)
            yb = yb.to(device)

            logits = model(xb)
            prob = torch.softmax(logits, dim=1) #squish logits and convert back to probabilites! Use softmax for multi-class prediction
            preds = torch.argmax(prob, dim=1)

            all_y.append(yb.cpu()); all_p.append(prob.cpu()); all_pred.append(preds.cpu())
            
        
        y = torch.cat(all_y).numpy()
        p = torch.cat(all_p).numpy()
        preds = torch.cat(all_pred).numpy()

        logger.info(f" ####### METRICS ######")
        #logger.info(f"Precision Recall curve: {precision_recall_curve(y, p)}")
        logger.info(f"Roc-auc-score: {roc_auc_score(y,p, multi_class='ovr', average='macro')}")
        logger.info(f"Confusion Matrix: {confusion_matrix( y, preds, labels=[0,1])}")
        logger.info(f"Classification report: \n{classification_report(y, preds, zero_division=0)}")


def main():

    input_dim = 4
    base_model = linear_mlp(input_dim)

    #train 
    #model, best_model_state = quick_train(base_model, x_train_torch, y_train_torch, n_factors=30, n_epochs=20, batch_size=512, device="mps")

    #save the best model state
    #torch.save(best_model_state, "iris_model_weights.pkl")

    #test
    #eval = evaluate(base_model, x_test_torch, y_test_torch, batch_size= 512)

if __name__=="__main__":
    main()
