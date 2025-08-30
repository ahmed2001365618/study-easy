import cv2
import mediapipe as mp
import pyautogui
from collections import deque
import time
import requests
import base64
import threading
import numpy as np   
import keyboard

w, h = 166, 166         
fps = 30                
confirm_frames = 6       
OPENROUTER_API_KEY = "your-api-key"
OPENROUTER_MODEL = "model-name"
ANALYZE_PROMPT = "summary in english the scientific content in the video"
LOG_FILE = "analysis_log.txt"


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, fps)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


is_playing = True
away_count = 0
look_count = 0


times = deque(maxlen=500)
tick = cv2.getTickCount
freq = cv2.getTickFrequency()

def toggle_vlc():
   
    
    keyboard.send("space")


def analyze_frame():
   
    def worker():
        
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        
        ts = time.strftime("%Y%m%d-%H%M%S")
        img_filename = f"desktop_{ts}.jpg"
        cv2.imwrite(img_filename, frame)
        print(f"[+] Desktop screenshot saved: {img_filename}")

        
        _, buffer = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ANALYZE_PROMPT},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                    ]
                }
            ]
        }

        try:
            resp = requests.post("URL-HERE",
                                 headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]

          
            if isinstance(content, str):
                analysis = content
            elif isinstance(content, list):
                analysis = "\n".join(
                    block.get("text", "") for block in content if block.get("type") == "text"
                )
            else:
                analysis = "[Unexpected content format]"

            if not analysis.strip():
                analysis = "[No text returned]"

            # Print
            print("\n=== OpenRouter Analysis ===")
            print(analysis)
            print("===========================\n")

            # Save to log file
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(f"[Screenshot: {img_filename}]\n")
                f.write(analysis + "\n")

        except Exception as e:
            print("Error analyzing frame:", e)

    threading.Thread(target=worker, daemon=True).start()



while True:
    ok, frame = cap.read()
    if not ok:
        break

    ih, iw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    looking_center = False
    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        # Head yaw
        x_left_outer  = lms[33].x * iw
        x_right_outer = lms[263].x * iw
        face_w = max(abs(x_right_outer - x_left_outer), 1e-6)

        x_mid_eyes = 0.5 * (x_left_outer + x_right_outer)
        x_nose = lms[1].x * iw
        yaw = (x_nose - x_mid_eyes) / face_w

        # Relaxed threshold
        looking_center = abs(yaw) < 0.25
    else:
        looking_center = False

    
    if looking_center:
        look_count += 1
        away_count = 0
        if not is_playing and look_count >= confirm_frames:
            toggle_vlc()
            is_playing = True
    else:
        away_count += 1
        look_count = 0
        if is_playing and away_count >= confirm_frames:
            toggle_vlc()
            is_playing = False
           
            analyze_frame()

 
    display = frame.copy()
    status_text = "Playing" if is_playing else "Paused & Analyzing"
    status_color = (235, 23, 23) if is_playing else (235, 23, 23)
    # cv2.rectangle(display, (0, 0), (iw, 40), (0, 0, 0), -1)
    cv2.putText(display, status_text, (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)

  
    # times.append(tick() / freq)
    # if len(times) >= 2:
    #     fps_est = int(round((len(times) - 1) / (times[-1] - times[0] + 1e-9)))
    #     cv2.putText(display, f"FPS ~ {fps_est}", (12, ih - 12),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Camera Feedback", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
