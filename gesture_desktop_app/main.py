import asyncio
import edge_tts
import pygame
import os
import tempfile
import cv2
import mediapipe as mp
import pyttsx3
import threading
import queue
import time

VOICE_ENABLED = True
speech_queue = queue.Queue()

GESTURE_MESSAGES = {
    "OPEN PALM": "Halo!",
    "FOUR FINGERS": "Saya adalah.",
    "THREE FINGERS": "Mahasiswa.",
    "ONE FINGER": "politeknik negeri bali, dan saya afrizal, "
}



async def speak_indonesian(text):
    voice = "id-ID-GadisNeural"

    with tempfile.NamedTemporaryFile(
        suffix=".mp3", delete=False
    ) as f:
        audio_path = f.name

    try:
        communicate = edge_tts.Communicate(
            text, voice
        )
        await communicate.save(audio_path)

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

        pygame.mixer.music.unload()

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def speech_worker():
    while True:
        message = speech_queue.get()

        try:
            if message is None:
                break

            asyncio.run(speak_indonesian(message))

        except Exception as exc:
            print("Kesalahan suara:", exc)

        finally:
            speech_queue.task_done()


    if indonesian_voice:
        engine.setProperty("voice", indonesian_voice.id)
        print("Suara bahasa Indonesia:", indonesian_voice.name)
    else:
        print("PERINGATAN: Suara bahasa Indonesia belum ditemukan.")
        print("Suara bawaan Windows akan digunakan.")
        print("Daftar suara yang tersedia:")

        for voice in voices:
            print("-", voice.name, "|", voice.id)

    while True:
        message = speech_queue.get()

        try:
            if message is None:
                break

            engine.say(message)
            engine.runAndWait()

        except Exception as exc:
            print("Kesalahan suara:", exc)

        finally:
            speech_queue.task_done()

    engine.stop()


threading.Thread(
    target=speech_worker,
    daemon=True
).start()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark

    # Deteksi sederhana untuk jempol.
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    thumb_up = (
        abs(thumb_tip.x - thumb_mcp.x)
        > abs(thumb_ip.x - thumb_mcp.x)
    )

    # Deteksi empat jari lainnya.
    other_fingers = [
        lm[8].y < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y,
    ]

    return thumb_up, other_fingers


def classify_gesture(thumb_up, other_fingers):
    count = sum(other_fingers) + int(thumb_up)

    if count == 5:
        return "OPEN PALM", count

    if thumb_up and not any(other_fingers):
        return "THUMBS UP", count

    if count == 0:
        return "FIST", count

    if other_fingers == [
        True, True, False, False
    ] and not thumb_up:
        return "PEACE", count

    if count == 3:
        return "THREE FINGERS", count

    if count == 4:
        return "FOUR FINGERS", count

    if count == 1:
        return "ONE FINGER", count

    return f"{count} FINGERS", count


def main():
    global VOICE_ENABLED

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Kamera tidak ditemukan.")
        return

    last_spoken_gesture = None
    candidate_gesture = None
    candidate_since = time.time()
    last_spoken_time = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ok, frame = camera.read()

            if not ok:
                print("Gagal membaca kamera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            result = hands.process(rgb)
            gesture = "NO HAND"
            finger_count = 0

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                thumb_up, other_fingers = fingers_up(hand)

                gesture, finger_count = classify_gesture(
                    thumb_up, other_fingers
                )

                now = time.time()

                if gesture != candidate_gesture:
                    candidate_gesture = gesture
                    candidate_since = now

                stable = now - candidate_since >= 0.6
                cooldown = now - last_spoken_time >= 2

                if (
                    stable
                    and gesture in GESTURE_MESSAGES
                    and gesture != last_spoken_gesture
                    and cooldown
                    and VOICE_ENABLED
                ):
                    speech_queue.put(
                        GESTURE_MESSAGES[gesture]
                    )
                    last_spoken_gesture = gesture
                    last_spoken_time = now

                if (
                    stable
                    and gesture not in GESTURE_MESSAGES
                ):
                    last_spoken_gesture = None

            else:
                candidate_gesture = None
                last_spoken_gesture = None

            # Panel informasi.
            cv2.rectangle(
                frame, (10, 10), (460, 115),
                (25, 25, 25), -1
            )

            cv2.putText(
                frame, f"Gesture: {gesture}",
                (22, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (80, 240, 150), 2
            )

            cv2.putText(
                frame, f"Jari: {finger_count}",
                (22, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            voice_label = "ON" if VOICE_ENABLED else "OFF"

            cv2.putText(
                frame,
                f"Suara: {voice_label} | Q: keluar | S: suara",
                (22, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1
            )

            cv2.imshow(
                "Gesture Recognition - Hello!",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                VOICE_ENABLED = not VOICE_ENABLED
                print(
                    "Suara:",
                    "ON" if VOICE_ENABLED else "OFF"
                )

    camera.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)


if __name__ == "__main__":
    main()
