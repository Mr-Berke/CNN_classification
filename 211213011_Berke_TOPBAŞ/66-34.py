import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from star_preprocessing import function 

epochs = 50
batch_size = 16

# Veriyi yükleme ve hazırlama
x, y = function()
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

for i in range(5): 
    # Eğitim ve test bölmeleri
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=i)
    
    # Model tanımlama
    input_size = x_train.shape[1]
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_size,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    
    # Modeli derleme
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    
    # Eğitim ve test kaybı/doğruluk listeleri
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # Model eğitimi
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=batch_size,
            verbose=0
        )
        
        # Eğitim sonuçlarını kaydetme
        train_loss = history.history["loss"][0]
        train_accuracy = history.history["accuracy"][0]
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Test sonuçlarını değerlendirme
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Her epoch sonunda konsola yazdır
        print(f"Epoch {epoch + 1}/{epochs}, \n"
              f"Eğitim Kaybı: {train_loss:.4f}, Eğitim Doğruluğu: {train_accuracy:.4f},\n"
              f"Test Kaybı: {test_loss:.4f}, Test Doğruluğu: {test_accuracy:.4f}\n")

    # Test verileri üzerinde tahmin
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
   
    # Konfüzyon Matrisi
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Konfüzyon Matrisi (Test Seti):\n{conf_matrix}\n")
    print(f"Test Doğruluk Oranı: {accuracy:.4f}\n")

    # Karışıklık matrisini görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.title(f"Konfüzyon Matrisi\nDoğruluk={accuracy:.4f}")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.show()

    # Eğitim ve test kaybı/doğruluk grafikleri
    plt.figure(figsize=(12, 6))

    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Eğitim Kaybı", color="blue")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Kaybı", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.title("Eğitim ve Test Kaybı")
    plt.legend()

    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Eğitim Doğruluğu", color="blue")
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Doğruluğu", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk")
    plt.title("Eğitim ve Test Doğruluğu")
    plt.legend()

    plt.tight_layout()
    plt.show()
