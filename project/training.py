# ─────────────────────────────────────────────────────────
# training.py
# Zuständig für:
# → Modell bauen
# → Modell trainieren
# → Modell speichern
# → Ergebnisse visualisieren
# ─────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle


# ══════════════════════════════════════════════════════════
# MODELL BAUEN
# ══════════════════════════════════════════════════════════

def modell_bauen(anzahl_klassen):
    """
    Baut ein CNN Modell mit Transfer Learning
    Basis: MobileNetV2 (schnell & effizient)
    """

    print(f"\n🧠 Baue Modell für {anzahl_klassen} Klassen...")

    # ─── Basis Modell (vortrainiert) ──────────────────────
    basis = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    basis.trainable = False  # Erst einfrieren

    # ─── Augmentierung ────────────────────────────────────
    augmentierung = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        # ❌ KEIN RandomFlip! (Zeichensprache spiegelverkehrt)
    ], name="augmentierung")

    # ─── Komplettes Modell ────────────────────────────────
    eingabe = tf.keras.Input(shape=(224, 224, 3))
    x       = augmentierung(eingabe)
    x       = tf.keras.layers.Rescaling(1./255)(x)
    x       = basis(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.4)(x)
    x       = tf.keras.layers.Dense(128, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    ausgabe = tf.keras.layers.Dense(
                  anzahl_klassen,
                  activation="softmax")(x)

    modell = tf.keras.Model(eingabe, ausgabe)

    modell.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("✅ Modell gebaut!")
    modell.summary()
    return modell


def koordinaten_modell_bauen(anzahl_klassen):
    """
    Baut ein einfaches Modell für MediaPipe Koordinaten
    Schneller zu trainieren als CNN
    """

    print(f"\n🧠 Baue Koordinaten-Modell...")

    modell = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(anzahl_klassen, activation="softmax")
    ])

    modell.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modell


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

def foto_modell_trainieren(dataset_ordner="daten/dataset",
                           modell_pfad="daten/modelle/foto_modell.h5",
                           epochen=30):
    """
    Trainiert CNN Modell direkt auf Fotos
    """

    print("\n🚀 Starte Foto-Training...")
    os.makedirs(os.path.dirname(modell_pfad), exist_ok=True)

    # ─── Daten laden ──────────────────────────────────────
    train_daten = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_ordner}/train",
        image_size=(224, 224),
        batch_size=32,
        label_mode="categorical",
        shuffle=True
    )

    val_daten = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_ordner}/val",
        image_size=(224, 224),
        batch_size=32,
        label_mode="categorical",
        shuffle=False
    )

    klassen        = train_daten.class_names
    anzahl_klassen = len(klassen)
    print(f"✅ Klassen: {klassen}")

    # Performance optimieren
    train_daten = train_daten.cache().prefetch(tf.data.AUTOTUNE)
    val_daten   = val_daten.cache().prefetch(tf.data.AUTOTUNE)

    # ─── Modell ───────────────────────────────────────────
    modell = modell_bauen(anzahl_klassen)

    # ─── Callbacks ────────────────────────────────────────
    callbacks = [
        # Bestes Modell speichern
        tf.keras.callbacks.ModelCheckpoint(
            modell_pfad,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1
        ),
        # Training stoppen wenn keine Verbesserung
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Lernrate reduzieren
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    # ─── Phase 1: Basis eingefroren ───────────────────────
    print("\n📌 Phase 1: Transfer Learning...")
    history1 = modell.fit(
        train_daten,
        validation_data=val_daten,
        epochs=epochen,
        callbacks=callbacks
    )

    # ─── Phase 2: Fine-Tuning ─────────────────────────────
    print("\n📌 Phase 2: Fine-Tuning...")

    # Obere Layer des Basis-Modells auftauen
    basis_modell = modell.layers[3]
    basis_modell.trainable = True

    # Nur letzte 30 Layer trainieren
    for layer in basis_modell.layers[:-30]:
        layer.trainable = False

    # Kleinere Lernrate für Fine-Tuning
    modell.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = modell.fit(
        train_daten,
        validation_data=val_daten,
        epochs=10,
        callbacks=callbacks
    )

    # ─── Klassen speichern ────────────────────────────────
    klassen_pfad = modell_pfad.replace(".h5", "_klassen.npy")
    np.save(klassen_pfad, klassen)

    # ─── Ergebnisse ───────────────────────────────────────
    beste_genauigkeit = max(history1.history["val_accuracy"] +
                           history2.history["val_accuracy"])
    print(f"\n✅ Training abgeschlossen!")
    print(f"   Beste Genauigkeit: {beste_genauigkeit:.2%}")

    ergebnis_visualisieren(history1, history2)
    return modell, klassen


def koordinaten_modell_trainieren(
        koordinaten_csv="daten/koordinaten.csv",
        modell_pfad="daten/modelle/koordinaten_modell.h5",
        epochen=50):
    """
    Trainiert Modell auf MediaPipe Koordinaten
    Schneller & braucht weniger Daten
    """

    print("\n🚀 Starte Koordinaten-Training...")
    os.makedirs(os.path.dirname(modell_pfad), exist_ok=True)

    # ─── Daten laden ──────────────────────────────────────
    daten = pd.read_csv(koordinaten_csv)
    X     = daten.drop("label", axis=1).values
    y_raw = daten["label"].values

    # Labels zu Zahlen
    encoder = LabelEncoder()
    y       = encoder.fit_transform(y_raw)

    # Encoder speichern
    encoder_pfad = modell_pfad.replace(".h5", "_encoder.pkl")
    with open(encoder_pfad, "wb") as f:
        pickle.dump(encoder, f)

    # Aufteilen
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    anzahl_klassen = len(encoder.classes_)
    print(f"✅ Klassen: {list(encoder.classes_)}")
    print(f"   Training:   {len(X_train)} Samples")
    print(f"   Validierung:{len(X_val)} Samples")

    # ─── Modell ───────────────────────────────────────────
    modell = koordinaten_modell_bauen(anzahl_klassen)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            modell_pfad,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5,
            factor=0.5
        )
    ]

    # ─── Training ─────────────────────────────────────────
    history = modell.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochen,
        batch_size=32,
        callbacks=callbacks
    )

    beste = max(history.history["val_accuracy"])
    print(f"\n✅ Training abgeschlossen!")
    print(f"   Beste Genauigkeit: {beste:.2%}")

    return modell, encoder


# ══════════════════════════════════════════════════════════
# VISUALISIERUNG
# ══════════════════════════════════════════════════════════

def ergebnis_visualisieren(history1, history2=None):
    """Erstellt Trainings-Grafiken"""

    # Historien zusammenführen
    if history2:
        genauigkeit = (history1.history["accuracy"] +
                      history2.history["accuracy"])
        val_genauigkeit = (history1.history["val_accuracy"] +
                          history2.history["val_accuracy"])
        verlust = (history1.history["loss"] +
                  history2.history["loss"])
        val_verlust = (history1.history["val_loss"] +
                      history2.history["val_loss"])
    else:
        genauigkeit     = history1.history["accuracy"]
        val_genauigkeit = history1.history["val_accuracy"]
        verlust         = history1.history["loss"]
        val_verlust     = history1.history["val_loss"]

    epochen = range(1, len(genauigkeit) + 1)

    plt.figure(figsize=(12, 4))

    # Genauigkeit
    plt.subplot(1, 2, 1)
    plt.plot(epochen, genauigkeit,     "b-", label="Training")
    plt.plot(epochen, val_genauigkeit, "r-", label="Validierung")
    plt.title("Genauigkeit")
    plt.xlabel("Epoche")
    plt.ylabel("Genauigkeit")
    plt.legend()
    plt.grid(True)

    # Verlust
    plt.subplot(1, 2, 2)
    plt.plot(epochen, verlust,     "b-", label="Training")
    plt.plot(epochen, val_verlust, "r-", label="Validierung")
    plt.title("Verlust")
    plt.xlabel("Epoche")
    plt.ylabel("Verlust")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("daten/training_ergebnis.png")
    plt.show()
    print("✅ Grafik gespeichert: daten/training_ergebnis.png")
echtzeit.py