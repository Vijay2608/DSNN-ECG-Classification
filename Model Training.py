import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ECG_Dataset import ECGDataset, ECGTransform
from Model_DSNN import DSNN
from Preprocess import extract_heartbeats, create_shifted

BATCH_SIZE = 64

def focal_loss(inputs, targets, alpha=1.0, gamma=2.0):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def export_for_hardware(model, filename):
    weights = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            if v.dtype in [torch.qint8, torch.quint8]:
                weights[k] = v.dequantize().cpu().numpy()
            else:
                weights[k] = v.detach().cpu().numpy()
    np.savez(filename, **weights)

def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        X, y = extract_heartbeats('mitbih_data')
        print(f"Data shapes: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s = create_shifted(X_train)
    X_train = np.concatenate([X_train, X_train_s])
    y_train = np.concatenate([y_train, y_train])
    train_dataset = ECGDataset(X_train, y_train, transform=ECGTransform())
    test_dataset = ECGDataset(X_test, y_test)
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    class_weights[3] *= 2
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = DSNN().to(device)
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(
            quant_min=0, quant_max=127, dtype=torch.qint8
        ),
        weight=torch.quantization.MinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8
        )
    )
    model.qconfig = qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    best_f1 = 0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    patience = 5
    epochs_no_improve = 0
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        all_train_preds, all_train_true = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x.unsqueeze(1).float())
            loss = focal_loss(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_true.extend(y.cpu().numpy())
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracy = accuracy_score(all_train_true, all_train_preds)
        train_accuracies.append(train_accuracy)
        model.eval()
        epoch_test_loss = 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x.unsqueeze(1).float())
                loss = focal_loss(outputs, y)
                epoch_test_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.cpu().numpy())
        test_losses.append(epoch_test_loss / len(test_loader))
        f1 = f1_score(all_true, all_preds, average='macro')
        accuracy = accuracy_score(all_true, all_preds)
        test_accuracies.append(accuracy)
        scheduler.step(f1)
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Train Accuracy = {train_accuracy:.4f}, Validation Loss = {test_losses[-1]:.4f}, Validation Accuracy = {accuracy:.4f}, Macro F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x.unsqueeze(1).float())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())
    print("\nClassification Report:")
    print(classification_report(
        all_true, all_preds,
        target_names=['Normal', 'LBBB', 'RBBB', 'APC', 'PVC', 'Paced']
    ))
    total_accuracy = accuracy_score(all_true, all_preds) * 100
    print(f"\nTotal Accuracy: {total_accuracy:.2f}%")
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)
    class_names = ['Normal', 'LBBB', 'RBBB', 'APC', 'PVC', 'Paced']
    plot_confusion_matrix(all_true, all_preds, class_names)
    quantized_model = torch.quantization.convert(model.eval(), inplace=False)
    export_for_hardware(quantized_model, "dsnn_quant_weights.npz")

if __name__ == "__main__":
    main()