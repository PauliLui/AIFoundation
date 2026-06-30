# ─────────────────────────────────────────────────────────
# main.py
# Hauptprogramm - Hier startet alles
# ─────────────────────────────────────────────────────────

import os
import datensatz
from training import (
    foto_modell_trainieren,
    koordinaten_modell_trainieren
)
from echtzeit import ZeichenspracheErkennung


def menu():
    """Hauptmenü"""

    print("\n" + "═" * 50)
    print("   🤟 ZEICHENSPRACHE KI SYSTEM")
    print("═" * 50)
    print("1. ZIP Dateien verarbeiten")
    print("2. Datensatz aufteilen (Train/Val)")
    print("3. Datensatz Statistik anzeigen")
    print("4. MediaPipe Koordinaten extrahieren")
    print("5. Koordinaten-Modell trainieren ⚡ (Empfohlen)")
    print("6. Foto-Modell trainieren 📷 (Langsamer)")
    print("7. Echtzeit Erkennung starten 🎥")
    print("8. Komplett-Setup (1-5 automatisch)")
    print("0. Beenden")
    print("─" * 50)

    return input("Auswahl: ").strip()


def komplett_setup():
    """
    Führt alle Schritte automatisch durch:
    ZIP → Datensatz → Koordinaten → Training
    """

    print("\n🚀 Starte Komplett-Setup...")
    print("─" * 50)

    # ─── Schritt 1: ZIPs verarbeiten ──────────────────────
    print("\n[1/4] ZIP Dateien verarbeiten...")
    if not datensatz.alle_zips_verarbeiten():
        print("\n⚠️  Bitte erst ZIP Dateien in")
        print("   'daten/rohdaten/' ablegen!")
        return

    # ─── Schritt 2: Aufteilen ─────────────────────────────
    print("\n[2/4] Datensatz aufteilen...")

    # Entpackten Ordner finden
    entpackt_ordner = "daten/entpackt"
    quell_ordner    = None

    for ordner in os.listdir(entpackt_ordner):
        pfad = os.path.join(entpackt_ordner, ordner)
        if os.path.isdir(pfad):
            quell_ordner = pfad
            break

    if not quell_ordner:
        print("❌ Kein entpackter Ordner gefunden!")
        return

    datensatz.datensatz_aufteilen(quell_ordner)

    # ─── Schritt 3: Statistik ─────────────────────────────
    print("\n[3/4] Datensatz prüfen...")
    datensatz.datensatz_statistik()

    # ─── Schritt 4: Koordinaten + Training ────────────────
    print("\n[4/4] Koordinaten extrahieren & trainieren...")
    datensatz.koordinaten_extrahieren()
    koordinaten_modell_trainieren()

    print("\n" + "═" * 50)
    print("✅ SETUP ABGESCHLOSSEN!")
    print("   Modell gespeichert in: daten/modelle/")
    print("   Jetzt Option 7 für Echtzeit-Erkennung!")
    print("═" * 50)


def main():
    """Hauptprogramm"""

    # Ordner erstellen
    os.makedirs("daten/rohdaten", exist_ok=True)
    os.makedirs("daten/modelle",  exist_ok=True)

    print("\n💡 Tipp: Lege deine ZIP Dateien in")
    print("         'daten/rohdaten/' und wähle Option 8")

    while True:
        auswahl = menu()

        if auswahl == "1":
            datensatz.alle_zips_verarbeiten()

        elif auswahl == "2":
            quell = input("Quell-Ordner Pfad: ").strip()
            datensatz.datensatz_aufteilen(quell)

        elif auswahl == "3":
            datensatz.datensatz_statistik()

        elif auswahl == "4":
            datensatz.koordinaten_extrahieren()

        elif auswahl == "5":
            koordinaten_modell_trainieren()

        elif auswahl == "6":
            foto_modell_trainieren()

        elif auswahl == "7":
            # Prüfen ob Modell existiert
            modell_pfad  = "daten/modelle/koordinaten_modell.h5"
            encoder_pfad = "daten/modelle/koordinaten_modell_encoder.pkl"

            if not os.path.exists(modell_pfad):
                print("❌ Kein Modell gefunden!")
                print("   Bitte erst trainieren (Option 5)")
            else:
                erkennung = ZeichenspracheErkennung(
                    modell_pfad,
                    encoder_pfad
                )
                erkennung.starten()

        elif auswahl == "8":
            komplett_setup()

        elif auswahl == "0":
            print("\n👋 Auf Wiedersehen!")
            break

        else:
            print("❌ Ungültige Auswahl!")


if __name__ == "__main__":
    main()
