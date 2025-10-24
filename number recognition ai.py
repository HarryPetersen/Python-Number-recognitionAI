import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from tkinter import *
import cv2
from PIL import Image, ImageGrab

def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train.reshape(-1,28,28,1), y_train, epochs=3, validation_data=(x_test.reshape(-1,28,28,1), y_test))

    model.save("digit_model.h5")
    print("✅ Model trained and saved as digit_model.h5")

def load_model():
    return keras.models.load_model("digit_model.h5")

class DrawApp:
    def __init__(self, model):
        self.model = model
        self.root = Tk()
        self.root.title("Draw a Digit (0–9)")
        self.root.geometry("400x480")
        self.root.configure(bg="white")

        self.canvas = Canvas(self.root, width=300, height=300, bg='white')
        self.canvas.pack(pady=20)

        self.canvas.bind('<B1-Motion>', self.draw_lines)

        Button(self.root, text="Predict", command=self.predict_digit, bg="lightgreen", font=("Arial", 14)).pack(pady=10)
        Button(self.root, text="Clear", command=self.clear_canvas, bg="lightcoral", font=("Arial", 14)).pack()

        self.label = Label(self.root, text="Draw a number above!", font=("Arial", 16), bg="white")
        self.label.pack(pady=10)

        self.root.mainloop()

    def draw_lines(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='black', outline='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a number above!")

    def predict_digit(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
        img = img.resize((28,28))
        img = np.array(img)
        img = 255 - img
        img = img / 255.0
        img = img.reshape(1,28,28,1)

        pred = np.argmax(self.model.predict(img))
        self.label.config(text=f"Prediction: {pred}")

if __name__ == "__main__":
    import os
    if not os.path.exists("digit_model.h5"):
        train_model()

    model = load_model()
    DrawApp(model)
