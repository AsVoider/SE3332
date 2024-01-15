import matplotlib.pyplot as plt


def visualize_history(history, metrics=['acc', 'loss']):
    if 'acc' in metrics:
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], label='train_acc', c='r')
        plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label='val_acc', c='g')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('acc.png')

    if 'loss' in metrics:
        plt.subplot(122)
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='train_loss', c='r')
        plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='val_loss', c='g')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('acc.png')
