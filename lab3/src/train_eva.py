import torch
from tqdm import tqdm
import lab3_data_scratch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_size = (128, 128)  # You can adjust this to balance speed and accuracy
transform = A.Compose(
    [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(),
        ToTensorV2()
    ]
)


def evaluate(model, val_data, loss_fn, weights=None, device='cpu', verbose=0):
    # set device
    if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif isinstance(device, torch.device):
        device = device
    else:
        device = torch.device('cpu')

    model = model.to(device)

    if weights:
        model.load_state_dict(torch.load(weights))
        print('Weights loaded successfully from path:', weights)
        print('====================================================')

    with torch.no_grad():
        model.eval()
        val_correct = 0
        val_total = len(val_data) * val_data.batch_size
        running_loss = 0.
        if verbose == 1:
            val_data = tqdm(val_data, desc='Evaluate: ', ncols=100)
        for data_batch, label_batch in val_data:
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)

            output_batch = model(data_batch)

            loss = loss_fn(output_batch, label_batch.long())
            running_loss += loss.item()

            _, predicted_labels = torch.max(output_batch.data, dim=1)

            val_correct += (label_batch == predicted_labels).sum().item()
        val_loss = running_loss / len(val_data)
        val_acc = val_correct / val_total
        return val_loss, val_acc


def train(model, train_data, loss_fn, optimizer, epochs, weights=None, save_last_weights_path=None,
          save_best_weights_path=None, freeze=False, steps_per_epoch=None, save_acc_path=None,
          device='cpu', validation_data=None, validation_split=None, scheduler=None):
    assert not (validation_data is not None and validation_split is not None)

    if weights:
        model.load_state_dict(torch.load(weights))
        print('Weights loaded successfully from path:', weights)
        print('====================================================')

    # set device
    if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif isinstance(device, torch.device):
        device = device
    else:
        device = torch.device('cpu')

    if validation_data is not None:
        val_data = validation_data
    elif validation_split is not None:
        train_data, val_data = lab3_data_scratch.split_dataloader(train_data, 0.2)
    else:
        val_data = None

    # save best model
    if save_best_weights_path:
        if val_data is None:
            train_data, val_data = lab3_data_scratch.split_dataloader(train_data, 0.2)
        best_loss, _ = evaluate(model, val_data, device=device, loss_fn=loss_fn)
    best_train_acc = 0
    best_val_acc = 0

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)

    num_steps = len(train_data)
    iterator = iter(train_data)
    count_steps = 1

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }

    # add model to device
    model = model.to(device)

    ############################### Train and Val ##########################################
    for epoch in range(1, epochs + 1):
        running_loss = 0.
        train_correct = 0
        train_total = steps_per_epoch * train_data.batch_size

        model.train()

        for step in tqdm(range(steps_per_epoch), desc=f'epoch: {epoch}/{epochs}: ', ncols=100):
            img_batch, label_batch = next(iterator)
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()

            output_batch = model(img_batch)

            loss = loss_fn(output_batch, label_batch.long())

            loss.backward()

            optimizer.step()

            _, predicted_labels = torch.max(output_batch.data, dim=1)

            train_correct += (label_batch == predicted_labels).sum().item()

            running_loss += loss.item()

            if count_steps == num_steps:
                count_steps = 0
                iterator = iter(train_data)
            count_steps += 1

        train_loss = running_loss / steps_per_epoch
        train_accuracy = train_correct / train_total

        # reduce lr
        if scheduler:
            scheduler.step(train_loss)

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_accuracy))

        if val_data is not None:
            val_loss, val_acc = evaluate(model, val_data, device=device, loss_fn=loss_fn)
            print(
                f'epoch: {epoch}, train_accuracy: {train_accuracy: .2f}, loss: {train_loss: .3f}, val_accuracy: {val_acc: .2f}, val_loss: {val_loss:.3f}')

            if save_best_weights_path:
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), save_best_weights_path)
                    print(f'Saved successfully best weights to:', save_best_weights_path)
            print(save_acc_path, )
            if save_acc_path:
                if train_accuracy > best_train_acc and val_acc > best_val_acc:
                    best_train_acc = train_accuracy
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), save_acc_path)
                    print(f'Saved successfully best acc weights to', save_acc_path)
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
        else:
            print(f'epoch: {epoch}, train_accuracy: {train_accuracy: .2f}, loss: {train_loss: .3f}')
    if save_last_weights_path:
        torch.save(model.state_dict(), save_last_weights_path)
        print(f'Saved successfully last weights to:', save_last_weights_path)
    return model, history
