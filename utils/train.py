def train_contrastive(rank, world_size, args, model, loss_fn, train_loader, val_loader, optimizer, num_epochs, log_dir='runs/experiment', model_save_path='contrastive_model.pth'):
    # Initialize the process group for distributed training
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Wrap the model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[rank])
    
    # Create a SummaryWriter only for the master process
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        # Set up the DistributedSampler to shuffle the data properly
        train_loader.sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Rank {rank}")):
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

        if rank == 0:  # Log only from the master process
            writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Rank {rank} Validation"):
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

        if rank == 0:  # Log only from the master process
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

    if rank == 0:
        writer.close()
        # Save the model after the last epoch
        torch.save(model.module.state_dict(), model_save_path)  # .module to save the underlying model
        print(f"Model saved to {model_save_path}")

    torch.distributed.destroy_process_group()  # Clean up the process group

def train_baseline(rank, world_size, args, model, train_loader, val_loader, optimizer, criterion, num_epochs, log_dir='runs/baseline', model_save_path='baseline_model.pth'):
    # Initialize the process group for distributed training
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Wrap the model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[rank])

    # Create a SummaryWriter only for the master process
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        # Set up the DistributedSampler to shuffle the data properly
        train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Rank {rank}")):
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

        if rank == 0:  # Log only from the master process
            writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Rank {rank} Validation"):
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

        if rank == 0:  # Log only from the master process
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

    if rank == 0:
        writer.close()
        # Save the model after the last epoch
        torch.save(model.module.state_dict(), model_save_path)  # .module to save the underlying model
        print(f"Model saved to {model_save_path}")

    torch.distributed.destroy_process_group()  # Clean up the process group


