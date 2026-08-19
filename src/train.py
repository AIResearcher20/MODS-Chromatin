import torch
import torch.nn as nn
import time

def train_model(model, train_loader, val_loader, epochs=150, lr=0.0005, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f" Training on {device}...")
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            y = batch.y.view(-1, 4)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * batch.num_graphs
        
        train_loss = total_train / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                y = batch.y.view(-1, 4)
                loss = criterion(pred, y)
                total_val += loss.item() * batch.num_graphs
        
        val_loss = total_val / len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_gat.pth')
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    print(f"\n Training complete in {(time.time()-start)/60:.2f} min")
    print(f"   Best val loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses
