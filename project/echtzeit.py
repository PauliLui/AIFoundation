# ─────────────────────────────────────────────────────────
# echtzeit.py
# Zuständig für:
# → Kamera öffnen
# → Hand mit MediaPipe erkennen
# → Zeichen klassifizieren
# → Text ausgeben
# ─────────────────────────────────────────────────────────

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import time


class ZeichenspracheErkennung:

    def __init__(self,
                 modell_pfad="daten/modelle/koordinaten_modell.h5",
                 encoder_pfad="daten/modelle/koordinaten_modell_encoder.pkl"):

        print("\n🤖 Lade Zeichensprache System...")

        # ─── Modell laden ─────────────────────────────────
        self.modell = tf.keras.models.load_model(modell_pfad)

        # ─── Encoder laden ────────────────────────────────
        with open(encoder_pfad, "rb") as f:
            self.encoder = pickle.load(f)

        self.klassen = self.encoder.classes_
        print(f"✅ Klassen geladen: {self.klassen}")

        # ─── MediaPipe Setup ──────────────────────────────
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # ─── Text Sammler ─────────────────────────────────
        self.erkannter_text  = ""
        self.letztes_zeichen = ""
        self.zeichen_zaehler = 0
        self.benoetigt       = 15   # Wie oft gleich = sicher
        self.min_sicherheit  = 0.85 # Mindest-Konfidenz

        print("✅ System bereit!\n")


    def hand_koordinaten(self, hand_landmarks):
        """Extrahiert 63 Koordinaten aus MediaPipe Landmarks"""
        koordinaten = []
        for punkt in hand_landmarks.landmark:
            koordinaten.extend([punkt.x, punkt.y, punkt.z])
        return np.array(koordinaten).reshape(1, -1)


    def zeichen_erkennen(self, koordinaten):
        """Gibt erkanntes Zeichen und Sicherheit zurück"""
        vorhersage = self.modell.predict(koordinaten, verbose=0)
        sicherheit = float(np.max(vorhersage))
        index      = np.argmax(vorhersage)
        zeichen    = self.klassen[index]
        return zeichen, sicherheit


    def text_aktualisieren(self, zeichen, sicherheit):
        """
        Fügt Zeichen zum Text hinzu wenn:
        → Sicherheit hoch genug
        → Zeichen stabil (mehrmals hintereinander gleich)
        """
        hinzugefuegt = False

        if sicherheit >= self.min_sicherheit:
            if zeichen == self.letztes_zeichen:
                self.zeichen_zaehler += 1
            else:
                self.zeichen_zaehler = 0
                self.letztes_zeichen = zeichen

            # Stabil genug?
            if self.zeichen_zaehler >= self.benoetigt:
                self.erkannter_text  += zeichen
                self.zeichen_zaehler  = 0
                hinzugefuegt          = True
                print(f"✓ Zeichen erkannt: {zeichen}"
                      f" | Text: {self.erkannter_text}")

        return hinzugefuegt


    def starten(self):
        """Startet die Echtzeit Erkennung"""

        print("📷 Starte Kamera...")
        print("   Steuerung:")
        print("   LEERTASTE → Leerzeichen")
        print("   BACKSPACE → Letztes Zeichen löschen")
        print("   C         → Text löschen")
        print("   Q         → Beenden\n")

        kamera = cv2.VideoCapture(0)

        if not kamera.isOpened():
            print("❌ Kamera nicht gefunden!")
            return

        while True:
            ret, frame = kamera.read()
            if not ret:
                break

            # Spiegeln (natürlicher für Nutzer)
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ─── Hand erkennen ────────────────────────────
            results = self.hands.process(rgb)

            zeichen    = ""
            sicherheit = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # Hand zeichnen
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(
                            color=(0,255,0), thickness=2),
                        self.mp_draw.DrawingSpec(
                            color=(0,0,255), thickness=2)
                    )

                    # Koordinaten & Erkennung
                    koordinaten         = self.hand_koordinaten(
                                              hand_landmarks)
                    zeichen, sicherheit = self.zeichen_erkennen(
                                              koordinaten)

                    # Text aktualisieren
                    self.text_aktualisieren(zeichen, sicherheit)

            # ─── Anzeige ──────────────────────────────────
            self.anzeige_zeichnen(frame, zeichen, sicherheit)

            cv2.imshow("Zeichensprache Erkennung", frame)

            # ─── Tastatur ─────────────────────────────────
            taste = cv2.waitKey(1) & 0xFF

            if taste == ord("q"):
                break
            elif taste == ord(" "):
                self.erkannter_text += " "
            elif taste == 8:  # Backspace
                self.erkannter_text = self.erkannter_text[:-1]
            elif taste == ord("c"):
                self.erkannter_text = ""
                print("Text gelöscht")

        kamera.release()
        cv2.destroyAllWindows()
        print(f"\n📝 Finaler Text: {self.erkannter_text}")


    def anzeige_zeichnen(self, frame, zeichen, sicherheit):
        """Zeichnet alle Informationen auf das Bild"""

        h, w = frame.shape[:2]

        # Hintergrund für Text
        cv2.rectangle(frame, (0, 0), (w, 120),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h-60), (w, h),
                      (0, 0, 0), -1)

        # Aktuelles Zeichen
        farbe = (0, 255, 0) if sicherheit >= self.min_sicherheit \
                else (0, 165, 255)

        cv2.putText(frame, f"Zeichen: {zeichen}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, farbe, 2)

        # Sicherheit als Balken
        if zeichen:
            balken_breite = int(sicherheit * 300)
            cv2.rectangle(frame, (10, 50),
                          (10 + balken_breite, 70), farbe, -1)
            cv2.rectangle(frame, (10, 50), (310, 70),
                          (255,255,255), 1)
            cv2.putText(frame, f"{sicherheit:.0%}",
                        (320, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,255), 1)

        # Stabilitäts-Fortschritt
        if self.zeichen_zaehler > 0:
            fortschritt = int(
                (self.zeichen_zaehler / self.benoetigt) * 200)
            cv2.rectangle(frame, (10, 80),
                          (10 + fortschritt, 95),
                          (255, 255, 0), -1)
            cv2.putText(frame, "Stabilität:",
                        (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255,255,0), 1)

        # Erkannter Text
        anzeige_text = self.erkannter_text[-40:]  # Letzte 40 Zeichen
        cv2.putText(frame, f"Text: {anzeige_text}",
                    (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # Steuerung
        cv2.putText(frame,
                    "SPACE=Leerzeichen | C=Loeschen | Q=Beenden",
                    (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1)
main.py