from flask import Flask, render_template, request, redirect, Response, jsonify
import os, cv2, numpy as np

# FIX matplotlib crash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#memory veriable
last_names = {}
stable_names = {}
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
DATASET_FOLDER = "dataset"
USER_FILE = "users.txt"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

live_info = []

# ---------- AUTH ---------- #

def save_user(email, password):
    with open(USER_FILE, "a") as f:
        f.write(f"{email},{password}\n")

def check_user(email, password):
    if not os.path.exists(USER_FILE):
        return False
    with open(USER_FILE, "r") as f:
        for line in f:
            e, p = line.strip().split(",")
            if e == email and p == password:
                return True
    return False

@app.route('/')
def auth():
    return render_template("auth.html")

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form['email']
    password = request.form['password']

    if not email.endswith("@gmail.com"):
        return "Use Gmail only"

    save_user(email, password)
    return redirect('/')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    if check_user(email, password):
        return redirect('/choice')
    return "Invalid login"

# ---------- CHOICE ---------- #

@app.route('/choice')
def choice():
    return render_template("choice.html")

@app.route('/go_label')
def go_label():
    return redirect('/label')

@app.route('/go_main')
def go_main():
    return redirect('/main')

# ---------- MODEL ---------- #

def load_model():
    if os.path.exists("model.yml") and os.path.exists("labels.npy"):
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read("model.yml")
        label_map = np.load("labels.npy", allow_pickle=True).item()
        return model, label_map
    return None, {}

def recognize(face, model, label_map):
    if model is None:
        return "Unknown"
    label, confidence = model.predict(face)
    if confidence > 85:
        return "Unknown"
    return label_map.get(label, "Unknown")

# ---------- DETECTION ---------- #

def detect_faces(gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40,40))
    return sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[:15]

# ---------- GRAPH ---------- #

def generate_graph(known, unknown):
    plt.figure(figsize=(4,4))
    plt.pie([known, unknown], labels=['Known', 'Unknown'], autopct='%1.1f%%')
    path = "static/uploads/graph.png"
    plt.savefig(path)
    plt.close()
    return path

# ---------- ROUTES ---------- #

@app.route('/main')
def main():
    return render_template("index.html")

@app.route('/label')
def label():
    return render_template("label.html")

# ---------- UPLOAD ---------- #

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    model, labels = load_model()

    info = []

    for i, (x, y, w, h) in enumerate(faces):
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (200,200))
        name = recognize(face_crop, model, labels)

        # Draw box
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        info.append(f"Face{i+1}: {name} (x={x}, y={y})")

    # ✅ SAVE IMAGE WITH BOXES
    cv2.imwrite(path, img)

    # ✅ GRAPH (DO NOT TOUCH IMAGE AGAIN)
    known = sum(1 for i in info if "Unknown" not in i)
    unknown = len(info) - known

    import matplotlib.pyplot as plt
    plt.figure()
    plt.pie([known, unknown], labels=["Known", "Unknown"], autopct='%1.1f%%')

    graph_path = os.path.join('static/uploads', 'graph.png')
    plt.savefig(graph_path)
    plt.close()

    return render_template('index.html', image=path, info=info, graph=graph_path)

# ---------- CAMERA PREVIEW ---------- #

@app.route('/camera')
def camera_page():
    return render_template("camera.html")

def camera_stream():
    while True:
        ret, frame = camera.read()
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/camera_feed')
def camera_feed():
    return Response(camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- CAPTURE ---------- #

@app.route('/capture')
def capture():
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(gray)
    model, label_map = load_model()

    known, unknown = 0, 0
    info = []

    for i,(x,y,w,h) in enumerate(faces):
        face = cv2.resize(gray[y:y+h,x:x+w],(200,200))
        name = recognize(face, model, label_map)

        if name=="Unknown":
            color=(0,0,255)
            unknown+=1
        else:
            color=(0,255,0)
            known+=1

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        info.append(f"Face{i+1}: {name}, Coordinates: (x={x}, y={y})")

    graph = generate_graph(known, unknown)

    path="static/uploads/capture.jpg"
    cv2.imwrite(path,frame)
    # Count known vs unknown
    known = sum(1 for i in info if "Unknown" not in i)
    unknown = len(info) - known

    # Create graph
    labels = ['Known', 'Unknown']
    values = [known, unknown]

    plt.figure()
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    graph_path = os.path.join('static/uploads', 'graph.png')
    plt.savefig(graph_path)
    plt.close()

    return render_template('index.html', image=path, info=info, graph=graph_path)

# ---------- LIVE ---------- #

def generate_frames():
    global live_info
    model, label_map = load_model()

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray)
        temp = []

        for i,(x,y,w,h) in enumerate(faces):
            face = cv2.resize(gray[y:y+h,x:x+w],(200,200))
            name = recognize(face, model, label_map)

            # 🔥 Stability key (based on position)
            key = (x//50, y//50)

            if key not in last_names:
                last_names[key] = []
                stable_names[key] = "Unknown"

            last_names[key].append(name)

            if len(last_names[key]) > 5:
                last_names[key].pop(0)

            final_name = max(set(last_names[key]), key=last_names[key].count)

            if last_names[key].count(final_name) >= 3:
                stable_names[key] = final_name

            name = stable_names[key]

            # 🎨 Color
            color = (0,255,0) if name!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            

            temp.append(f"Face{i+1}: {name}, Coordinates: (x={x}, y={y})")

        live_info = temp

        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/live')
def live():
    return render_template("live.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_data')
def live_data():
    return jsonify({"data": live_info})

# ---------- RUN ---------- #

if __name__ == "__main__":
    app.run(debug=True)