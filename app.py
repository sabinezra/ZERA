"""
ZERA - Robotic Assistant System
Flask Backend: Serial communication, YOLOv8 follow, Reminders, Voice command processing
"""

from flask import Flask, render_template, Response, request, jsonify, stream_with_context
import threading
import time
import datetime
import json
import queue

app = Flask(__name__)

# ── Serial Configuration ──
ser = None
robot_speed = 150
SERIAL_AVAILABLE = False
serial_lock = threading.Lock()

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print("[WARN] pyserial not installed. Serial features disabled.")

def get_serial_ports():
    if not SERIAL_AVAILABLE:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]

def connect_serial(port, baudrate=9600):
    global ser
    try:
        with serial_lock:
            if ser and ser.is_open:
                ser.close()
            ser = serial.Serial(port, baudrate, timeout=1)
        return True, f"Connected to {port} at {baudrate} baud"
    except Exception as e:
        return False, str(e)

def send_command(cmd):
    global ser
    with serial_lock:
        if ser and ser.is_open:
            try:
                ser.write((cmd + "\n").encode())
                return True, f"Sent: {cmd}"
            except Exception as e:
                return False, str(e)
    return False, "Arduino not connected"

def get_serial_status():
    with serial_lock:
        if ser and ser.is_open:
            return True, ser.port
    return False, None

# ── Reminder System ──
reminders = []
reminder_queue = queue.Queue()
reminder_lock = threading.Lock()

def add_reminder(date_str, time_str, message):
    # Safety check for empty values
    if not time_str:
        time_str = "00:00"
    else:
        time_str = str(time_str)[:5]  # forces "14:30:00" to be "14:30"
        
    if not date_str:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    r = {"id": int(time.time() * 1000) % 100000, "datetime": dt, "message": message, "triggered": False}
    with reminder_lock:
        reminders.append(r)
    print(f"[SUCCESS] Reminder saved: {r['datetime']} - {r['message']}") # Prints to terminal
    return r

def check_reminders_loop():
    while True:
        now = datetime.datetime.now()
        with reminder_lock:
            for r in reminders:
                if not r["triggered"] and now >= r["datetime"]:
                    r["triggered"] = True
                    reminder_queue.put(r)
        time.sleep(1)

threading.Thread(target=check_reminders_loop, daemon=True).start()

# ── YOLOv8 Follow Mode ──
follow_active = False
follow_streaming = False
follow_lock = threading.Lock()
YOLO_AVAILABLE = False
yolo_model = None
yolo_model_lock = threading.Lock()

try:
    import cv2
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] opencv/ultralytics not installed. Follow mode disabled.")

def get_yolo_model():
    global yolo_model
    if not YOLO_AVAILABLE:
        return None
    with yolo_model_lock:
        if yolo_model is None:
            yolo_model = YOLO("yolov8n.pt")
    return yolo_model

def generate_follow_feed():
    global follow_active, follow_streaming
    model = get_yolo_model()
    if model is None:
        return
    
    with follow_lock:
        if follow_streaming:
            return
        follow_streaming = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with follow_lock:
            follow_streaming = False
        return
        
    last_cmd = None
    last_sent_at = 0.0
    try:
        while True:
            with follow_lock:
                if not follow_active:
                    break
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, classes=[0], verbose=False)
            cx_frame = frame.shape[1] // 2
            found = False
            target_x = cx_frame
            target_area = 0
            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = max(1, (x2 - x1) * (y2 - y1))
                    if area > target_area:
                        found = True
                        target_area = area
                        target_x = (x1 + x2) // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 2)
                    cv2.putText(frame, "HUMAN", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
            cmd = "S:0"
            if found:
                diff = target_x - cx_frame
                if abs(diff) < 50:
                    cmd = f"F:{robot_speed}"
                elif diff > 0:
                    cmd = f"R:{max(80, robot_speed // 2)}"
                else:
                    cmd = f"L:{max(80, robot_speed // 2)}"
            now = time.monotonic()
            if cmd != last_cmd or now - last_sent_at >= 0.08:
                send_command(cmd)
                last_cmd = cmd
                last_sent_at = now
            cv2.line(frame, (cx_frame, 0), (cx_frame, frame.shape[0]), (0, 0, 255), 1)
            label = "FOLLOWING" if found else "SEARCHING..."
            color = (0, 255, 136) if found else (0, 100, 255)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            _, buf = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.03)
    finally:
        cap.release()
        with follow_lock:
            follow_active = False
            follow_streaming = False
        send_command("S:0")

# ── Voice Command Processing ──
import re

def process_voice_command(text):
    global robot_speed, follow_active, ser
    t = text.lower().strip()

    def robot_move(action, command, ok_text):
        ok, msg = send_command(command)
        if ok:
            return {"action": action, "response": ok_text, "success": True}
        return {"action": "chat", "response": f"Cannot move robot: {msg}", "success": False}
    
    # Music
    if any(w in t for w in ["music", "spotify", "song", "songs", "play music"]):
        return {"action": "open_music", "response": "Opening Spotify for you."}
    # Video
    if any(w in t for w in ["video", "youtube", "watch video", "play video"]):
        return {"action": "open_video", "response": "Opening YouTube for you."}
    # Camera
    if any(w in t for w in ["camera", "cam", "webcam", "open camera"]):
        return {"action": "open_camera", "response": "Opening camera feed."}
    # Forward
    if any(w in t for w in ["forward", "ahead", "go forward", "move forward"]):
        return robot_move("robot_forward", f"F:{robot_speed}", f"Moving forward at speed {robot_speed}.")
    # Backward
    if any(w in t for w in ["backward", "back", "reverse", "go back"]):
        return robot_move("robot_backward", f"B:{robot_speed}", f"Moving backward at speed {robot_speed}.")
    # Left
    if "left" in t and ("turn" in t or "go" in t):
        return robot_move("robot_left", f"L:{robot_speed}", "Turning left.")
    # Right
    if "right" in t and ("turn" in t or "go" in t):
        return robot_move("robot_right", f"R:{robot_speed}", "Turning right.")
    # Stop
    if any(w in t for w in ["stop", "halt", "brake", "stop robot"]):
        ok, msg = send_command("S:0")
        if ok:
            return {"action": "robot_stop", "response": "Robot stopped.", "success": True}
        return {"action": "chat", "response": f"Cannot stop robot: {msg}", "success": False}
    # Speed
    if "speed" in t:
        nums = re.findall(r"\d+", t)
        if nums:
            robot_speed = max(0, min(255, int(nums[0])))
            return {"action": "set_speed", "response": f"Speed set to {robot_speed}.", "speed": robot_speed}
    # Follow
    if any(w in t for w in ["follow", "follow me", "start follow", "track", "tracking"]):
        if not YOLO_AVAILABLE:
            return {"action": "chat", "response": "Follow mode is unavailable. Install opencv-python and ultralytics first."}
        with follow_lock:
            follow_active = True
        return {"action": "start_follow", "response": "Follow mode activated. Detecting humans with YOLOv8."}
    if any(w in t for w in ["stop follow", "stop tracking", "cancel follow"]):
        with follow_lock:
            follow_active = False
        send_command("S:0")
        return {"action": "stop_follow", "response": "Follow mode deactivated."}
    # Reminder
    if any(w in t for w in ["remind", "reminder", "alarm", "alert", "set reminder"]):
        return {"action": "open_reminder", "response": "Opening reminder settings. Set the date, time, and message."}
    # Connect
    if any(w in t for w in ["connect", "arduino", "serial"]):
        ports = get_serial_ports()
        if ports:
            ok, msg = connect_serial(ports[0])
            connected, port = get_serial_status()
            return {"action": "serial_connect", "response": msg, "success": ok, "connected": connected, "port": port}
        return {"action": "serial_connect", "response": "No serial ports detected. Check USB connection."}
    # Disconnect
    if "disconnect" in t:
        with serial_lock:
            if ser and ser.is_open:
                ser.close()
                return {"action": "serial_disconnect", "response": "Disconnected from Arduino.", "success": True, "connected": False}
        return {"action": "chat", "response": "Not connected to any device."}
    # Greetings
    if any(w in t for w in ["hello", "hi ", "hey", "greetings", "good morning", "good evening"]):
        return {"action": "chat", "response": "Hello! I am ZERA, your robotic assistant. How can I help you today?"}
    if any(w in t for w in ["who are you", "your name", "what are you"]):
        return {"action": "chat", "response": "I am ZERA, an advanced robotic assistant. I can play music, stream videos, control movement, follow humans using AI, and manage reminders. Say 'help' for all commands."}
    if any(w in t for w in ["thank", "thanks"]):
        return {"action": "chat", "response": "You're welcome! Always ready to serve."}
    if any(w in t for w in ["bye", "goodbye", "see you"]):
        return {"action": "chat", "response": "Goodbye! ZERA standing by."}
    if any(w in t for w in ["help", "what can you do", "commands", "list"]):
        return {"action": "chat", "response": "Available commands: Open Music, Open Video, Open Camera, Move Forward/Backward/Left/Right, Stop, Set Speed [0-255], Start Follow, Stop Follow, Set Reminder, Connect Arduino, Disconnect. Try saying any of these!"}
    if any(w in t for w in ["time", "what time"]):
        now = datetime.datetime.now().strftime("%I:%M %p")
        return {"action": "chat", "response": f"Current time is {now}."}
    if any(w in t for w in ["date", "what date", "today"]):
        today = datetime.datetime.now().strftime("%B %d, %Y")
        return {"action": "chat", "response": f"Today is {today}."}
    return {"action": "chat", "response": f"I heard: '{text}'. I don't understand that command yet. Say 'help' to see what I can do."}

# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/serial/ports")
def api_serial_ports():
    return jsonify(get_serial_ports())

@app.route("/api/serial/connect", methods=["POST"])
def api_serial_connect():
    d = request.get_json(silent=True) or {}
    try:
        baudrate = int(d.get("baudrate", 9600))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Baudrate must be an integer"}), 400
    ok, msg = connect_serial(d.get("port", ""), baudrate)
    connected, current_port = get_serial_status()
    return jsonify({"success": ok, "message": msg, "connected": connected, "port": current_port})

@app.route("/api/serial/disconnect", methods=["POST"])
def api_serial_disconnect():
    global ser
    with serial_lock:
        if ser and ser.is_open:
            ser.close()
            return jsonify({"success": True, "message": "Disconnected", "connected": False})
    return jsonify({"success": False, "message": "Not connected", "connected": False})

@app.route("/api/serial/status")
def api_serial_status():
    with serial_lock:
        if ser and ser.is_open:
            return jsonify({"connected": True, "port": ser.port})
    return jsonify({"connected": False})

@app.route("/api/robot/command", methods=["POST"])
def api_robot_command():
    d = request.get_json(silent=True) or {}
    ok, msg = send_command(d.get("command", ""))
    return jsonify({"success": ok, "message": msg, "command": d.get("command", "")})

@app.route("/api/robot/speed", methods=["POST"])
def api_robot_speed():
    global robot_speed
    d = request.get_json(silent=True) or {}
    try:
        speed = int(d.get("speed", 150))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Speed must be an integer"}), 400
    robot_speed = max(0, min(255, speed))
    return jsonify({"success": True, "speed": robot_speed})

@app.route("/api/follow/start", methods=["POST"])
def api_follow_start():
    global follow_active
    if not YOLO_AVAILABLE:
        return jsonify({"success": False, "message": "Follow mode requires opencv-python and ultralytics"}), 503
    cam_probe = cv2.VideoCapture(0)
    if not cam_probe.isOpened():
        cam_probe.release()
        return jsonify({"success": False, "message": "Camera is not available for follow mode"}), 503
    cam_probe.release()
    with follow_lock:
        follow_active = True
    return jsonify({"success": True, "message": "Follow mode started", "active": True})

@app.route("/api/follow/stop", methods=["POST"])
def api_follow_stop():
    global follow_active
    with follow_lock:
        follow_active = False
    send_command("S:0")
    return jsonify({"success": True, "message": "Follow mode stopped", "active": False})

@app.route("/follow/feed")
def follow_feed():
    if not YOLO_AVAILABLE:
        return Response("Follow mode unavailable", status=503, mimetype="text/plain")
    with follow_lock:
        if not follow_active:
            return Response("Follow mode not active", status=403, mimetype="text/plain")
    return Response(stream_with_context(generate_follow_feed()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/reminders", methods=["GET"])
def api_get_reminders():
    with reminder_lock:
        return jsonify([{"id": r["id"], "datetime": r["datetime"].strftime("%Y-%m-%d %H:%M"),
                         "message": r["message"], "triggered": r["triggered"]} for r in reminders])

@app.route("/api/reminders", methods=["POST"])
def api_add_reminder():
    d = request.get_json(silent=True) or {}
    print(f"[DEBUG] Received data: {d}") # Prints to terminal
    
    try:
        date_val = d.get("date")
        time_val = d.get("time")
        msg_val = d.get("message")
        
        if not date_val or not time_val or not msg_val:
            print("[ERROR] Missing fields!")
            return jsonify({"success": False, "message": "Missing date, time, or message"}), 400
            
        r = add_reminder(date_val, time_val, msg_val)
        return jsonify({"success": True, "reminder": {"id": r["id"],
            "datetime": r["datetime"].strftime("%Y-%m-%d %H:%M"), "message": r["message"]}})
    except Exception as e:
        print(f"[ERROR] Reminder failed: {e}") # Prints exact error to terminal
        return jsonify({"success": False, "message": str(e)}), 400

@app.route("/api/reminders/<int:rid>", methods=["DELETE"])
def api_delete_reminder(rid):
    with reminder_lock:
        reminders[:] = [r for r in reminders if r["id"] != rid]
    return jsonify({"success": True})

@app.route("/api/reminders/alerts")
def api_reminder_alerts():
    def gen():
        while not reminder_queue.empty():
            try:
                reminder_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                r = reminder_queue.get(timeout=30)
                payload = json.dumps({"id": r["id"], "message": r["message"]})
                yield f"data: {payload}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(stream_with_context(gen()), mimetype="text/event-stream")

@app.route("/api/voice/process", methods=["POST"])
def api_voice_process():
    d = request.get_json(silent=True) or {}
    result = process_voice_command(d.get("text", ""))
    return jsonify(result)

@app.route("/api/system/state")
def api_system_state():
    connected, port = get_serial_status()
    with follow_lock:
        follow_state = follow_active
    return jsonify({
        "serial": {"connected": connected, "port": port},
        "robot": {"speed": robot_speed},
        "follow": {"available": YOLO_AVAILABLE, "active": follow_state}
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  ZERA Robotic Assistant System")
    print("=" * 50)
    print(f"  Serial available: {SERIAL_AVAILABLE}")
    print(f"  YOLOv8 available: {YOLO_AVAILABLE}")
    print(f"  Running at: http://localhost:5000")
    print("=" * 50)
    # Set debug=True only for local development
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)