import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from ECG_Dataset import ECGDataset, ECGTransform
from Model_DSNN import DSNN
from preprocess import extract_heartbeats, create_shifted

BATCH_SIZE = 64

def export_for_hardware(model, filename):
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(filename, **weights)

def main():
    X, y = extract_heartbeats('mitbih_data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_s = create_shifted(X_train)
    X_train = np.concatenate([X_train, X_train_s])
    y_train = np.concatenate([y_train, y_train])

    train_dataset = ECGDataset(X_train, y_train, transform=ECGTransform())
    test_dataset = ECGDataset(X_test, y_test)

    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DSNN()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_f1 = 0
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x.unsqueeze(1).float())
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x.unsqueeze(1).float())
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.cpu().numpy())

        f1 = f1_score(all_true, all_preds, average='macro')
        scheduler.step(f1)
        print(f"Epoch {epoch+1}: Macro F1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        all_preds, all_true = [], []
        for x, y in test_loader:
            outputs = model(x.unsqueeze(1).float())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=['Normal', 'LBBB', 'RBBB', 'APC', 'PVC', 'Paced']))

    quantized_model = torch.quantization.convert(model.eval(), inplace=False)
    export_for_hardware(quantized_model, "dsnn_quant_weights.npz")

if __name__ == "__main__":
    main()
