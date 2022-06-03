def train_epochs(model, loss_func, optimizer, device, train_dl, valid_dl, test_dl, 
                 formatting_func, epochs, start=0, scheduler=None, 
                 save_model_name=None, model_dir=None, tracking=False):

    # make directory for the model
    if model_dir and not os.path.isdir(model_dir): os.mkdir(model_dir)

    logs = []
    
    if tracking:   
        wandb.watch(model, log_freq=100)
        
    for epoch_idx in range(start, epochs+start):
                
        print("Epoch: {}/{}".format(epoch_idx+1, epochs+start))
        
        avg_train_loss, avg_train_acc = train(model, loss_func, optimizer, device, train_dl, 
                                              formatting_func, scheduler, tracking)
        print("Training Loss: {:.4f}, Training Accuracy: {:.4f}%".format(avg_train_loss, avg_train_acc*100))
        
        avg_valid_loss, avg_valid_acc = valid(model, loss_func, device, valid_dl, 
                                              formatting_func, tracking)
        print("Validation Loss: {:.4f}, Validation Accuracy: {:.4f}%".format(avg_valid_loss, avg_valid_acc*100))

        if test_dl:
            avg_test_loss, avg_test_acc = valid(model, loss_func, device, test_dl, 
                                                formatting_func, tracking)
            print("Test Loss: {:.4f}, Test Accuracy: {:.4f}%".format(avg_test_loss, avg_test_acc*100))
          
        metrics = {"epoch": epoch_idx,
                    "train_loss": avg_train_loss, 
                    "valid_loss": avg_valid_loss,
                    "train_acc": avg_train_acc, 
                    "valid_acc": avg_valid_acc} 

        if test_dl:
            metrics["test_loss"]=avg_test_loss
            metrics["test_acc"]=avg_test_acc 
                  
        logs.append(metrics)
                  
        if tracking:   
            wandb.log(metrics)
                
        if save_model_name != None:
            torch.save(model, os.path.join(model_dir, save_model_name+'-epoch-{}.pt'.format(epoch_idx+1)))
                
    return logs


def train(model, loss_func, optimizer, device, train_dl, formatting_func, scheduler=None, 
          tracking=False):
                  
    model.train()
    train_data_size = len(train_dl.dataset)
    train_loss = 0.0
    train_acc = 0.0

    for i, batch in enumerate(tqdm(train_dl)):
        inputs = batch[0].to(device).float()
        labels = batch[1].to(device).long()
        model = model.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_func(outputs, labels) 
        train_loss += loss.item() * inputs.size(0)
        loss.backward()

        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * inputs.size(0)
            
        optimizer.step()
        if scheduler:
            scheduler.step()

    avg_train_loss = train_loss/train_data_size 
    avg_train_acc = train_acc/float(train_data_size)

    return avg_train_loss, avg_train_acc


def valid(model, loss_func, device, valid_dl, formatting_func, 
          tracking=False):
                  
    model.eval()
    valid_data_size = len(valid_dl.dataset)
    valid_loss = 0.0
    valid_acc = 0.0
    
    start_time = time.time()
        
    with torch.no_grad():


        for j, batch in enumerate(tqdm(valid_dl)):
            
            inputs, labels = formatting_func(batch)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels) 
            valid_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs.size(0)     
            
    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc = valid_acc/float(valid_data_size)

