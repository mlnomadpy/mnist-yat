import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model_class, model_name, train_loader, val_loader, config, device, verbose=True):
    """Generic method to train any model class."""
    print(f"Training {model_name} on {device}...")
    model = model_class(dropout=config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config['lr'],
        momentum=0.9, weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    if verbose:
        print(f"Training {model_name}:")
        print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR")
        print("-" * 65)

    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f"{epoch+1:5d} | {history['train_loss'][-1]:10.4f} | "
                  f"{history['val_loss'][-1]:8.4f} | {history['train_acc'][-1]:9.4f} | "
                  f"{history['val_acc'][-1]:7.4f} | {current_lr:.6f}")
        scheduler.step()

        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                if verbose: print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
    return model, history
