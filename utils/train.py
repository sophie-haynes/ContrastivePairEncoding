import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def calculate_accuracy(outputs, labels):
    feats = outputs['feats']
    labels = outputs['labels']
    
    preds = torch.sign(feats.sum(dim=1))
    preds[preds == 0] = 1  # Handle the case where the sum is zero
    
    correct = (preds == labels.float()).sum().item()
    return correct / labels.size(0)

def train_contrastive(model, loss_fn, train_loader, val_loader, optimizer, num_epochs, device, log_dir='runs/experiment',model_save_path='contrastive_model.pth'):
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using GPU {device} for training!")

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(img1, img2, labels)

            loss_dict = loss_fn(outputs)
            loss = loss_dict['loss']

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            correct_train += calculate_accuracy(outputs, labels) * labels.size(0)
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_train_acc:.4f}")

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                img1, img2, labels = batch
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                outputs = model(img1, img2, labels)

                loss_dict = loss_fn(outputs)
                loss = loss_dict['loss']

                val_loss += loss.item()

                correct_val += calculate_accuracy(outputs, labels) * labels.size(0)
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

    writer.close()

    # Save the model after the last epoch
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def train_baseline(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, log_dir='runs/baseline', model_save_path='baseline_model.pth'):
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using GPU {device} for training!")

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            img1, _, labels = batch
            img1, labels = img1.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(img1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_train_acc:.4f}")

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                img1, _, labels = batch
                img1, labels = img1.to(device), labels.to(device)

                outputs = model(img1)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

    writer.close()

    # Save the model after the last epoch
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

