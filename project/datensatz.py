# ─────────────────────────────────────────────────────────
# datensatz.py
# Zuständig für:
# → ZIP entpacken
# → Ordnerstruktur prüfen
# → Koordinaten mit MediaPipe extrahieren
# → Trainingsdaten speichern

# ─────────────────────────────────────────────────────────

import os
import cv2
import zipfile
import shutil
import random
import numpy as np
import mediapipe as mp
import csv

# ─── MediaPipe Setup ──────────────────────────────────────
mp_hands    = mp.solutions.hands.Hands
mp_draw     = mp.solutions.drawing_utils
hand_detektor = mp_hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


# ══════════════════════════════════════════════════════════
# ZIP VERARBEITUNG
# ══════════════════════════════════════════════════════════

def zip_pruefen(zip_pfad):
    """Prüft ob ZIP Datei in Ordnung ist"""
    try:
        with zipfile.ZipFile(zip_pfad, "r") as zip_ref:
            ergebnis = zip_ref.testzip()
            if ergebnis is None:
                print(f"✅ ZIP ok: {zip_pfad}")
                return True
            else:
                print(f"❌ ZIP beschädigt bei: {ergebnis}")
                return False
    except zipfile.BadZipFile:
        print(f"❌ Keine gültige ZIP: {zip_pfad}")
        return False


def zip_struktur_anzeigen(zip_pfad):
    """Zeigt die ersten 20 Einträge der ZIP"""
    with zipfile.ZipFile(zip_pfad, "r") as zip_ref:
        eintraege = zip_ref.namelist()
        print(f"\n📁 ZIP Inhalt (erste 20 von {len(eintraege)}):")
        for eintrag in eintraege[:20]:
            print(f"   {eintrag}")


def zip_entpacken(zip_pfad, ziel_ordner):
    """
    Entpackt ZIP Datei mit Fortschrittsanzeige
    Überspringt bereits entpackte Dateien
    """

    # Schon entpackt?
    if os.path.exists(ziel_ordner) and os.listdir(ziel_ordner):
        print(f"✅ Bereits entpackt: {ziel_ordner}")
        return True

    if not zip_pruefen(zip_pfad):
        return False

    print(f"\n📦 Entpacke: {zip_pfad}")
    os.makedirs(ziel_ordner, exist_ok=True)

    with zipfile.ZipFile(zip_pfad, "r") as zip_ref:
        dateien = zip_ref.namelist()
        gesamt  = len(dateien)

        for i, datei in enumerate(dateien):
            zip_ref.extract(datei, ziel_ordner)

            # Fortschritt
            if i % 50 == 0:
                prozent = (i / gesamt) * 100
                balken  = "█" * int(prozent / 5)
                print(f"\r   [{balken:<20}] {prozent:.0f}%", end="")

    print(f"\r   [{'█' * 20}] 100%")
    print(f"✅ Entpackt nach: {ziel_ordner}\n")
    return True


def alle_zips_verarbeiten(rohdaten_ordner="daten/rohdaten",
                          ziel_ordner="daten/entpackt"):
    """
    Verarbeitet alle ZIP Dateien im rohdaten Ordner
    → Legt sie automatisch in den richtigen Ordner
    """

    os.makedirs(rohdaten_ordner, exist_ok=True)
    os.makedirs(ziel_ordner,    exist_ok=True)

    # Alle ZIPs finden
    zip_dateien = [
        f for f in os.listdir(rohdaten_ordner)
        if f.endswith(".zip")
    ]

    if not zip_dateien:
        print(f"⚠️  Keine ZIP Dateien in: {rohdaten_ordner}")
        print(f"    Bitte ZIP Dateien dort ablegen!")
        return False

    print(f"📦 {len(zip_dateien)} ZIP(s) gefunden:")

    for zip_datei in zip_dateien:
        zip_pfad    = os.path.join(rohdaten_ordner, zip_datei)
        ziel        = os.path.join(ziel_ordner,
                                   zip_datei.replace(".zip", ""))

        print(f"\n→ Verarbeite: {zip_datei}")
        zip_struktur_anzeigen(zip_pfad)
        zip_entpacken(zip_pfad, ziel)

    return True


# ══════════════════════════════════════════════════════════
# ORDNERSTRUKTUR
# ══════════════════════════════════════════════════════════

def datensatz_aufteilen(quell_ordner,
                        ziel_ordner="daten/dataset",
                        split=0.8):
    """
    Teilt Fotos automatisch in Train/Val auf
    80% Training, 20% Validierung
    """

    print(f"\n📂 Teile Datensatz auf...")
    gesamt_train = 0
    gesamt_val   = 0

    for klasse in sorted(os.listdir(quell_ordner)):
        klassen_pfad = os.path.join(quell_ordner, klasse)

        if not os.path.isdir(klassen_pfad):
            continue

        # Nur Bilder
        fotos = [
            f for f in os.listdir(klassen_pfad)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not fotos:
            continue

        # Mischen & aufteilen
        random.shuffle(fotos)
        grenze      = int(len(fotos) * split)
        train_fotos = fotos[:grenze]
        val_fotos   = fotos[grenze:]

        # Ordner erstellen
        train_pfad = os.path.join(ziel_ordner, "train", klasse)
        val_pfad   = os.path.join(ziel_ordner, "val",   klasse)
        os.makedirs(train_pfad, exist_ok=True)
        os.makedirs(val_pfad,   exist_ok=True)

        # Kopieren
        for foto in train_fotos:
            shutil.copy(
                os.path.join(klassen_pfad, foto),
                os.path.join(train_pfad,   foto)
            )

        for foto in val_fotos:
            shutil.copy(
                os.path.join(klassen_pfad, foto),
                os.path.join(val_pfad,     foto)
            )

        gesamt_train += len(train_fotos)
        gesamt_val   += len(val_fotos)

        print(f"   {klasse}: {len(train_fotos)} Train"
              f" | {len(val_fotos)} Val ✓")

    print(f"\n✅ Gesamt: {gesamt_train} Train"
          f" | {gesamt_val} Val")
    return True


def datensatz_statistik(dataset_ordner="daten/dataset"):
    """Zeigt detaillierte Statistik des Datensatzes"""

    print("\n📊 Datensatz Statistik:")
    print("─" * 50)

    for split in ["train", "val"]:
        split_pfad = os.path.join(dataset_ordner, split)

        if not os.path.exists(split_pfad):
            continue

        print(f"\n{split.upper()}:")
        gesamt = 0

        for klasse in sorted(os.listdir(split_pfad)):
            klassen_pfad = os.path.join(split_pfad, klasse)

            if not os.path.isdir(klassen_pfad):
                continue

            anzahl  = len(os.listdir(klassen_pfad))
            gesamt += anzahl
            balken  = "█" * (anzahl // 10)
            print(f"   {klasse:3s}: {anzahl:4d} Fotos  {balken}")

        print(f"   {'─' * 30}")
        print(f"   Gesamt: {gesamt} Fotos")


# ══════════════════════════════════════════════════════════
# MEDIAPIPE KOORDINATEN EXTRAHIEREN
# ══════════════════════════════════════════════════════════

def koordinaten_aus_foto(bild_pfad):
    """
    Extrahiert 63 Hand-Koordinaten aus einem Foto
    Gibt None zurück wenn keine Hand gefunden
    """

    bild = cv2.imread(bild_pfad)

    if bild is None:
        return None

    rgb     = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
    results = hand_detektor.process(rgb)

    if results.multi_hand_landmarks:
        koordinaten = []
        for punkt in results.multi_hand_landmarks[0].landmark:
            koordinaten.extend([punkt.x, punkt.y, punkt.z])
        return koordinaten  # 63 Zahlen

    return None  # Keine Hand gefunden


def koordinaten_extrahieren(dataset_ordner="daten/dataset",
                            ausgabe_datei="daten/koordinaten.csv"):
    """
    Verarbeitet alle Fotos und extrahiert MediaPipe Koordinaten
    Speichert alles in eine CSV Datei
    """

    print("\n🔍 Extrahiere MediaPipe Koordinaten...")

    os.makedirs(os.path.dirname(ausgabe_datei), exist_ok=True)

    erfolgreich = 0
    fehlgeschlagen = 0

    with open(ausgabe_datei, "w", newline="") as datei:
        writer = csv.writer(datei)

        # Header schreiben
        header = [f"p{i}" for i in range(63)] + ["label"]
        writer.writerow(header)

        for split in ["train", "val"]:
            split_pfad = os.path.join(dataset_ordner, split)

            if not os.path.exists(split_pfad):
                continue

            for klasse in sorted(os.listdir(split_pfad)):
                klassen_pfad = os.path.join(split_pfad, klasse)

                if not os.path.isdir(klassen_pfad):
                    continue

                fotos = [
                    f for f in os.listdir(klassen_pfad)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                for foto in fotos:
                    pfad        = os.path.join(klassen_pfad, foto)
                    koordinaten = koordinaten_aus_foto(pfad)

                    if koordinaten:
                        writer.writerow(koordinaten + [klasse])
                        erfolgreich += 1
                    else:
                        fehlgeschlagen += 1

                print(f"   ✓ {klasse} ({split}): "
                      f"{len(fotos)} Fotos verarbeitet")

    print(f"\n✅ Koordinaten extrahiert:")
    print(f"   Erfolgreich:    {erfolgreich}")
    print(f"   Fehlgeschlagen: {fehlgeschlagen}")
    print(f"   Gespeichert in: {ausgabe_datei}")

    return ausgabe_datei
training.py