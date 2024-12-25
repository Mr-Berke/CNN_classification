import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns 
from star_preprocessing import function
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam  

# Hedef değişkeni one-hot kodlama
X, y = function()
y = to_categorical(y, num_classes=6)

# Model oluşturma fonksiyonu
def build_model(input_dim, learning_rate):
    model = Sequential([
        Input(shape=(input_dim,)), 
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(6, activation="softmax")
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Modeli oluşturma ve eğitme
input_dim = X.shape[1]
model = build_model(input_dim=input_dim, learning_rate=0.001)
history = model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Model Performansı
y_pred = (model.predict(X) > 0.5).astype(int)
print(classification_report(y.argmax(axis=1), y_pred.argmax(axis=1)))

# Confusion Matrix ve Görselleştirme
def plot_confusion_matrix(y_true, y_pred):
    acc = round(accuracy_score(y_true, y_pred), 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues")
    plt.title('Konfüzyon Matris')
    plt.xlabel('Tahmin')
    plt.ylabel('Doğru')
    plt.title(f'Doğruluk: {acc}', size=10)
    plt.show()

y_true = y.argmax(axis=1)
y_pred_classes = y_pred.argmax(axis=1)
plot_confusion_matrix(y_true, y_pred_classes)

# Eğitim Sürecinin Görselleştirilmesi
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Doğruluk")
if "val_accuracy" in history.history:
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Doğruluk")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Kayıp")
if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Kayıp")
plt.show()
