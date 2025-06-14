from torch import nn, optim
from torch.utils.data import DataLoader
from models.transformer import HandwritingTransformer
from data.dataset import HandWritingDataset
from utils.params import Params
from utils.history import TrainingHistory
from training.early_stopping import EarlyStopping

def train(model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping, history):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history.add_loss(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs, val_targets = val_batch
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs.view(-1, val_outputs.size(-1)), val_targets.view(-1)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        history.add_val_loss(avg_val_loss)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def main():
    train_dataset = HandWritingDataset('data/train', transform=transform)
    val_dataset = HandWritingDataset('data/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = HandwritingTransformer(input_size=16 * 24, vocab_size=len(Params.vocab) + 2).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=5)
    history = TrainingHistory()

    train(model, train_loader, val_loader, criterion, optimizer, epochs=20, early_stopping=early_stopping, history=history)

if __name__ == "__main__":
    main()