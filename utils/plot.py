from matplotlib import pyplot as plt

def show_loss(train_loss, val_loss):
    epochs = list(range(1, len(train_loss) + 1))
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')

    min_train = min(train_loss)
    min_val = min(val_loss)
    idx_train = train_loss.index(min_train)
    idx_val = val_loss.index(min_val)

    plt.text(epochs[idx_train], min_train + 0.02, f"{min_train:.2f}", ha='center', fontsize=9, color='blue')
    plt.text(epochs[idx_val], min_val + 0.02, f"{min_val:.2f}", ha='center', fontsize=9, color='orange')

    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()


def show_acc(train_acc, val_acc):
    epochs = list(range(1, len(train_acc) + 1))
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_acc, label='Train Acc', marker='o')
    plt.plot(epochs, val_acc, label='Val Acc', marker='o')

    max_train = max(train_acc)
    max_val = max(val_acc)
    idx_train = train_acc.index(max_train)
    idx_val = val_acc.index(max_val)

    plt.text(epochs[idx_train], max_train + 0.5, f"{max_train:.2f}%", ha='center', fontsize=9, color='blue')
    plt.text(epochs[idx_val], max_val + 0.5, f"{max_val:.2f}%", ha='center', fontsize=9, color='orange')

    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()
