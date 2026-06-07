

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger, LambdaCallback
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy

# ─────────────────────────────────────────
# 0. REPRODUCIBILITY
# ─────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────
# 1. CONFIG  (edit these as needed)
# ─────────────────────────────────────────
CFG = dict(
    train_dir = "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    valid_dir = "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",

    img_size       = 380,          # EfficientNetB4 native size (better than 224)
    batch_size     = 16,           # lower if OOM; raise if GPU memory allows
    num_classes    = None,         # auto-detected from directory

    # --- Phase 1: frozen backbone ---
    phase1_epochs  = 10,
    phase1_lr      = 1e-3,

    # --- Phase 2: fine-tune top N layers of backbone ---
    phase2_epochs  = 30,
    phase2_lr      = 5e-5,
    unfreeze_layers= 50,           # last 50 backbone layers unfrozen

    label_smoothing= 0.1,
    dropout_rate   = 0.4,
    weight_decay   = 1e-4,

    mixup_alpha    = 0.2,          # 0 = disable MixUp
    cutmix_alpha   = 0.2,          # 0 = disable CutMix

    tta_steps      = 5,            # Test-Time Augmentation inference rounds
    patience       = 7,
    save_path      = "plant_disease_efficientb4.keras",
)

# ─────────────────────────────────────────
# 2. VERIFY PATHS
# ─────────────────────────────────────────
for key in ("train_dir", "valid_dir"):
    p = CFG[key]
    if not os.path.exists(p):
        raise FileNotFoundError(f"[ERROR] {key} not found: {p}")
    print(f"[OK] {key}: {p}")

# ─────────────────────────────────────────
# 3. tf.data PIPELINE  (faster than ImageDataGenerator)
# ─────────────────────────────────────────
IMG_SIZE  = CFG["img_size"]
AUTOTUNE  = tf.data.AUTOTUNE

def get_label_map(directory):
    classes = sorted([d for d in os.listdir(directory)
                      if os.path.isdir(os.path.join(directory, d))])
    return {c: i for i, c in enumerate(classes)}, classes

label_map, class_names = get_label_map(CFG["train_dir"])
num_classes = len(class_names)
CFG["num_classes"] = num_classes
print(f"[INFO] Detected {num_classes} classes.")

def load_dataset(directory, is_training=True):
    """Build a tf.data.Dataset from a directory of class sub-folders."""
    paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_map[class_name])

    paths  = tf.constant(paths)
    labels = tf.one_hot(labels, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: (load_and_preprocess(p, is_training), l),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(CFG["batch_size"], drop_remainder=is_training)
    if is_training:
        ds = ds.map(mixup_or_cutmix, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds, len(paths)

# ── Image loading & augmentation ──────────
@tf.function
def load_and_preprocess(path, augment):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_hue(img, 0.05)
        # Random rotation via contrib-free approach
        img = random_rotate(img)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img

@tf.function
def random_rotate(img, max_angle=25.0):
    angle = tf.random.uniform([], -max_angle, max_angle) * (np.pi / 180.0)
    img   = tf.keras.preprocessing.image.apply_affine_transform
    # Use tfa if available, otherwise skip rotation silently
    try:
        import tensorflow_addons as tfa
        img = tfa.image.rotate(img, angle, interpolation='BILINEAR')
    except ImportError:
        pass
    return img

# ── MixUp / CutMix ───────────────────────
def mixup(images, labels, alpha=CFG["mixup_alpha"]):
    batch = tf.shape(images)[0]
    lam   = tf.cast(
        tf.compat.v1.distributions.Beta(alpha, alpha).sample([batch, 1, 1, 1]),
        tf.float32)
    idx   = tf.random.shuffle(tf.range(batch))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, idx)
    lam2  = tf.squeeze(lam, axis=[1, 2, 3])[:, tf.newaxis]
    mixed_labels = lam2 * labels + (1 - lam2) * tf.gather(labels, idx)
    return mixed_images, mixed_labels

def rand_bbox(h, w, lam):
    cut_rat = tf.sqrt(1.0 - lam)
    cut_h   = tf.cast(tf.cast(h, tf.float32) * cut_rat, tf.int32)
    cut_w   = tf.cast(tf.cast(w, tf.float32) * cut_rat, tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    return y1, x1, y2, x2

def cutmix(images, labels, alpha=CFG["cutmix_alpha"]):
    batch = tf.shape(images)[0]
    h, w  = IMG_SIZE, IMG_SIZE
    lam   = tf.random.uniform([])
    y1, x1, y2, x2 = rand_bbox(h, w, lam)
    idx   = tf.random.shuffle(tf.range(batch))
    patch = tf.gather(images, idx)[:, y1:y2, x1:x2, :]

    # Compose via masking
    mask = tf.pad(
        tf.ones([batch, y2 - y1, x2 - x1, 1], tf.float32),
        [[0,0],[y1, h-(y2)],[x1, w-(x2)],[0,0]]
    )
    mixed = images * (1 - mask) + patch * mask
    lam_adj = 1.0 - tf.cast((y2-y1)*(x2-x1), tf.float32) / tf.cast(h*w, tf.float32)
    mixed_labels = lam_adj * labels + (1-lam_adj) * tf.gather(labels, idx)
    return mixed, mixed_labels

@tf.function
def mixup_or_cutmix(images, labels):
    """Randomly apply MixUp OR CutMix per batch."""
    if CFG["mixup_alpha"] > 0 and CFG["cutmix_alpha"] > 0:
        r = tf.random.uniform([])
        images, labels = tf.cond(
            r < 0.5,
            lambda: mixup(images, labels),
            lambda: cutmix(images, labels)
        )
    elif CFG["mixup_alpha"] > 0:
        images, labels = mixup(images, labels)
    elif CFG["cutmix_alpha"] > 0:
        images, labels = cutmix(images, labels)
    return images, labels

# ── Build datasets ────────────────────────
print("[INFO] Building train dataset...")
train_ds, n_train = load_dataset(CFG["train_dir"], is_training=True)
print("[INFO] Building validation dataset...")
valid_ds, n_valid = load_dataset(CFG["valid_dir"], is_training=False)

steps_per_epoch  = n_train // CFG["batch_size"]
validation_steps = n_valid // CFG["batch_size"]
print(f"[INFO] Train: {n_train} images | Valid: {n_valid} images")

# ─────────────────────────────────────────
# 4. CLASS WEIGHTS  (handle imbalanced data)
# ─────────────────────────────────────────
all_labels = []
for class_name in class_names:
    class_dir = os.path.join(CFG["train_dir"], class_name)
    if os.path.isdir(class_dir):
        n = len([f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
        all_labels.extend([label_map[class_name]] * n)

cw = compute_class_weight('balanced',
                           classes=np.arange(num_classes),
                           y=all_labels)
class_weight = {i: float(w) for i, w in enumerate(cw)}
print("[INFO] Class weights computed.")

# ─────────────────────────────────────────
# 5. MODEL ARCHITECTURE
# ─────────────────────────────────────────
def build_model(num_classes, img_size, dropout_rate):
    inp = Input(shape=(img_size, img_size, 3))

    # Backbone: EfficientNetB4 (much stronger than B0)
    backbone = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_tensor=inp
    )
    backbone.trainable = False   # frozen in phase 1

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(CFG["weight_decay"]))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(CFG["weight_decay"]))(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)
    return model, backbone

model, backbone = build_model(num_classes, IMG_SIZE, CFG["dropout_rate"])

# ─────────────────────────────────────────
# 6. COSINE ANNEALING LR SCHEDULE
# ─────────────────────────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        self.base_lr      = base_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        ws   = tf.cast(self.warmup_steps, tf.float32)
        ts   = tf.cast(self.total_steps,  tf.float32)

        warmup_lr = self.base_lr * step / ws
        cosine_lr = 0.5 * self.base_lr * (
            1 + tf.cos(np.pi * (step - ws) / (ts - ws)))
        return tf.where(step < ws, warmup_lr, cosine_lr)

    def get_config(self):
        return {"base_lr": self.base_lr,
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}

# ─────────────────────────────────────────
# 7. COMPILE HELPERS
# ─────────────────────────────────────────
loss_fn = CategoricalCrossentropy(label_smoothing=CFG["label_smoothing"])

def compile_phase(model, lr, total_steps, warmup_steps=0):
    if warmup_steps > 0:
        schedule = WarmupCosineDecay(lr, total_steps, warmup_steps)
    else:
        schedule = lr

    optimizer = AdamW(
        learning_rate=schedule,
        weight_decay=CFG["weight_decay"],
        clipnorm=1.0               # gradient clipping
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy',
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

# ─────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

checkpoint = ModelCheckpoint(
    CFG["save_path"],
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=CFG["patience"],
    restore_best_weights=True,
    verbose=1
)
csv_log = CSVLogger("logs/training_log.csv", append=True)

# LR tracker for plotting
lr_history = []
lr_tracker = LambdaCallback(
    on_epoch_end=lambda epoch, logs:
        lr_history.append(float(tf.keras.backend.get_value(
            model.optimizer.learning_rate)))
)

callbacks = [checkpoint, early_stop, csv_log, lr_tracker]

# ─────────────────────────────────────────
# 9. PHASE 1 — FROZEN BACKBONE
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  PHASE 1: Training classifier head (backbone frozen)")
print("="*55)

p1_steps = CFG["phase1_epochs"] * steps_per_epoch
compile_phase(model, CFG["phase1_lr"],
              total_steps=p1_steps, warmup_steps=p1_steps // 10)

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=CFG["phase1_epochs"],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# 10. PHASE 2 — FINE-TUNE TOP LAYERS
# ─────────────────────────────────────────
print("\n" + "="*55)
print(f"  PHASE 2: Fine-tuning last {CFG['unfreeze_layers']} backbone layers")
print("="*55)

# Unfreeze the top N layers of the backbone
backbone.trainable = True
for layer in backbone.layers[:-CFG["unfreeze_layers"]]:
    layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"[INFO] Trainable layers: {trainable}")

p2_steps = CFG["phase2_epochs"] * steps_per_epoch
compile_phase(model, CFG["phase2_lr"],
              total_steps=p2_steps, warmup_steps=p2_steps // 5)

history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=CFG["phase2_epochs"],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks,
    initial_epoch=CFG["phase1_epochs"],
    verbose=1
)

# ─────────────────────────────────────────
# 11. MERGE HISTORIES & SAVE
# ─────────────────────────────────────────
def merge(h1, h2):
    merged = {}
    for k in h1.history:
        merged[k] = h1.history[k] + h2.history.get(k, [])
    return merged

full_history = merge(history1, history2)
model.save(CFG["save_path"])
print(f"\n[SAVED] Model → {CFG['save_path']}")

with open("logs/history.json", "w") as f:
    json.dump(full_history, f)

# ─────────────────────────────────────────
# 12. TRAINING PLOTS
# ─────────────────────────────────────────
def plot_training(history, lr_hist, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["accuracy"]) + 1)
    phase_split = CFG["phase1_epochs"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Plant Disease Model — Training Diagnostics", fontsize=14, fontweight="bold")

    # Accuracy
    ax = axes[0]
    ax.plot(epochs, history["accuracy"],     label="Train Acc",  linewidth=2)
    ax.plot(epochs, history["val_accuracy"], label="Val Acc",    linewidth=2)
    ax.axvline(phase_split, color="gray", linestyle="--", alpha=0.7, label="Fine-tune start")
    ax.axhline(0.85, color="red", linestyle=":", alpha=0.6, label="85% target")
    ax.set_title("Accuracy"); ax.legend(); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(epochs, history["loss"],     label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss",   linewidth=2)
    ax.axvline(phase_split, color="gray", linestyle="--", alpha=0.7, label="Fine-tune start")
    ax.set_title("Loss"); ax.legend(); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    # LR
    ax = axes[2]
    ax.plot(range(1, len(lr_hist) + 1), lr_hist, color="darkorange", linewidth=2)
    ax.axvline(phase_split, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Learning Rate Schedule"); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] Plot → {path}")

plot_training(full_history, lr_history)

# ─────────────────────────────────────────
# 13. TEST-TIME AUGMENTATION (TTA) INFERENCE
# ─────────────────────────────────────────
def tta_predict(model, image_path, tta_steps=CFG["tta_steps"]):
    """
    Predict a single image using TTA (average over N augmented versions).
    image_path: str path to image file
    Returns: (predicted_class_name, confidence)
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    preds = []
    for _ in range(tta_steps):
        aug = tf.image.random_flip_left_right(img)
        aug = tf.image.random_brightness(aug, 0.15)
        aug = tf.image.random_contrast(aug, 0.85, 1.15)
        aug = tf.clip_by_value(aug, 0.0, 1.0)
        aug = tf.expand_dims(aug, 0)
        preds.append(model(aug, training=False).numpy())

    avg_pred  = np.mean(preds, axis=0)[0]
    class_idx = np.argmax(avg_pred)
    confidence = avg_pred[class_idx]
    return class_names[class_idx], float(confidence)


# ─────────────────────────────────────────
# 14. FINAL VALIDATION ACCURACY REPORT
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL EVALUATION")
print("="*55)
results = model.evaluate(valid_ds, steps=validation_steps, verbose=1)
metric_names = model.metrics_names
for name, val in zip(metric_names, results):
    print(f"  {name:20s}: {val:.4f}")

best_val_acc = max(full_history.get("val_accuracy", [0]))
print(f"\n  Best Val Accuracy : {best_val_acc*100:.2f}%")
if best_val_acc >= 0.85:
    print("  ✅ Target of 85%+ ACHIEVED!")
else:
    print("  ⚠️  Below 85% — consider more epochs or larger backbone (B5/B7).")

print("\n[DONE] Training complete.")
print(f"[INFO] Use tta_predict(model, 'path/to/image.jpg') for high-confidence inference.")