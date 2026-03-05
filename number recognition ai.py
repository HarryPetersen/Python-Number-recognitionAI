"""
Digit Draw + MNIST Trainer (Improved)
- Better CNN (2 conv blocks + dropout)
- Data augmentation (rotation/translation) built into the model
- tf.data pipeline (faster) + callbacks (early stop, LR reduce, checkpoint)
- Tkinter drawing uses an internal PIL image (no ImageGrab / no DPI issues)
- MNIST-like preprocessing: invert -> bbox crop -> pad -> resize -> center

Run:
  python digit_draw_app.py

If model doesn't exist, it trains and saves it.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import Tk, Canvas, Button, Label, Scale, HORIZONTAL, Frame
from PIL import Image, ImageDraw

tf.keras.utils.set_random_seed(42)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

MODEL_PATH = "digit_model.keras"


def build_model():
    aug = keras.Sequential(
        [
            layers.RandomTranslation(0.12, 0.12),
            layers.RandomRotation(0.08),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=(28, 28, 1))
    x = aug(inputs)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mnist_digit_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_datasets(batch_size=128):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


def train_model(save_path=MODEL_PATH):
    train_ds, test_ds = make_datasets()

    model = build_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=20,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(save_path)
    print(f"Model trained and saved to {save_path}")

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {acc:.4f}")


def load_model(path=MODEL_PATH):
    return keras.models.load_model(path)


def preprocess_pil_for_mnist(pil_img, out_size=28):
    """
    Convert user's drawing (large canvas) into MNIST-style 28x28 input.
    Steps:
      - grayscale -> invert (white background -> black becomes high)
      - find bounding box of ink
      - crop with margin, pad to square
      - resize ink region to ~20x20 then pad to 28x28
      - center using center-of-mass
    """
    img = np.array(pil_img).astype(np.uint8) 
    img = 255 - img 

    if img.max() < 10:
        return None

    ys, xs = np.where(img > 20)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    margin = 20
    y0 = max(0, y0 - margin)
    y1 = min(img.shape[0] - 1, y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(img.shape[1] - 1, x1 + margin)

    cropped = img[y0 : y1 + 1, x0 : x1 + 1]

    h, w = cropped.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off : y_off + h, x_off : x_off + w] = cropped
    square_pil = Image.fromarray(square)
    content_size = 20
    square_pil = square_pil.resize((content_size, content_size), Image.Resampling.LANCZOS)

    padded = Image.new("L", (out_size, out_size), 0)
    pad = (out_size - content_size) // 2
    padded.paste(square_pil, (pad, pad))

    arr = np.array(padded).astype(np.float32)

    total = arr.sum()
    if total > 0:
        ys, xs = np.indices(arr.shape)
        cy = (ys * arr).sum() / total
        cx = (xs * arr).sum() / total
        shift_y = int(round(arr.shape[0] / 2 - cy))
        shift_x = int(round(arr.shape[1] / 2 - cx))
        arr = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))

    arr = arr / 255.0
    return arr.reshape(1, out_size, out_size, 1)


class DrawApp:
    def __init__(self, model):
        self.model = model

        self.root = Tk()
        self.root.title("Draw a Digit (0–9)")
        self.root.geometry("420x540")
        self.root.configure(bg="white")

        self.canvas_size = 320
        self.canvas = Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white", highlightthickness=1)
        self.canvas.pack(pady=16)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        controls = Frame(self.root, bg="white")
        controls.pack(pady=6)

        self.brush = Scale(controls, from_=6, to=28, orient=HORIZONTAL, label="Brush", bg="white")
        self.brush.set(16)
        self.brush.pack(side="left", padx=10)

        Button(controls, text="Predict (P)", command=self.predict_digit, bg="lightgreen", font=("Arial", 12)).pack(side="left", padx=8)
        Button(controls, text="Clear (C)", command=self.clear_canvas, bg="lightcoral", font=("Arial", 12)).pack(side="left", padx=8)

        self.label = Label(self.root, text="Draw a number above!", font=("Arial", 16), bg="white")
        self.label.pack(pady=10)

        self.prob_label = Label(self.root, text="", font=("Arial", 12), bg="white", justify="left")
        self.prob_label.pack(pady=6)

        self.last_x = None
        self.last_y = None
        self.canvas.bind("<Button-1>", self.pen_down)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self.pen_up)

        self.root.bind("c", lambda e: self.clear_canvas())
        self.root.bind("C", lambda e: self.clear_canvas())
        self.root.bind("p", lambda e: self.predict_digit())
        self.root.bind("P", lambda e: self.predict_digit())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.root.mainloop()

    def pen_down(self, event):
        self.last_x, self.last_y = event.x, event.y

    def pen_up(self, event):
        self.last_x, self.last_y = None, None

    def draw_lines(self, event):
        x, y = event.x, event.y
        if self.last_x is None:
            self.last_x, self.last_y = x, y
            return

        w = int(self.brush.get())

        self.canvas.create_line(
            self.last_x, self.last_y, x, y,
            width=w, fill="black", capstyle="round", smooth=True, splinesteps=36
        )

        self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=w)

        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a number above!")
        self.prob_label.config(text="")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        x = preprocess_pil_for_mnist(self.image, out_size=28)
        if x is None:
            self.label.config(text="I don't see anything drawn 😅")
            self.prob_label.config(text="")
            return

        probs = self.model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))

        top3 = np.argsort(probs)[::-1][:3]
        top_text = "\n".join([f"{i}: {probs[i]*100:5.1f}%" for i in top3])

        self.label.config(text=f"Prediction: {pred}")
        self.prob_label.config(text=f"Top 3:\n{top_text}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model(MODEL_PATH)

    model = load_model(MODEL_PATH)
    DrawApp(model)