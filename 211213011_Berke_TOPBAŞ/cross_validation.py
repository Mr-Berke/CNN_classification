import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from star_preprocessing import function 

# K-Fold Cross-Validation
def cross_validate(model_fn, X, y, folds=5, epochs=50):
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    results, all_preds, all_labels = [], [], []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)
    num_classes = len(np.unique(y))

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Tüm fold'lar için kayıpları ve doğrulukları tutan liste
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1}/{folds}")

        model = model_fn(input_dim=X.shape[1], num_classes=num_classes, lr=0.001)
        history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat),
                            epochs=epochs, batch_size=16, verbose=0)

        # Eğitim ve test kayıplarını ve doğruluklarını al
        train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

        fold_metrics.append({
            "Fold": fold + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Test Loss": test_loss,
            "Test Accuracy": test_acc
        })

        # Test tahminleri
        y_pred = np.argmax(model.predict(X_test), axis=1)
        all_preds.extend(y_pred)
        all_labels.extend(y_test)

        accuracy = np.mean(y_pred == y_test)
        results.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    # Ortalama doğruluk
    mean_accuracy = np.mean(results)
    print(f"Ortalama Cross-Validation doğruluğu: {mean_accuracy:.4f}")
    
    # Confusion Matrix görselleştirme
    conf_matrix = confusion_matrix(all_labels, all_preds)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Konfüzyon matrisi bütün foldlar toplamı\n Ortalama Doğruluk: {mean_accuracy:.4f}")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.show()

    # Fold sonuçlarını DataFrame olarak döndür
    fold_metrics_df = pd.DataFrame(fold_metrics)
    print("\nFold Metrics:\n", fold_metrics_df)
    return fold_metrics_df

# Model ttanımlama
def create_model(input_dim, num_classes, lr):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

X, y = function()
print("5-Fold Cross-Validation")
results_5fold = cross_validate(create_model, X, y, folds=5, epochs=50)

print("\n10-Fold Cross-Validation")
results_10fold = cross_validate(create_model, X, y, folds=10, epochs=50)
