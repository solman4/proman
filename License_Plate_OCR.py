import cv2
import torch
import easyocr
import sqlite3
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from datetime import datetime
import threading
import logging
import os

# Thiết lập ghi log
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Khởi tạo Flask và Socket.IO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Thiết lập cơ sở dữ liệu
def init_db():
    conn = sqlite3.connect('vehicle_log.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS VehicleLogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            gate TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

# Khởi tạo các biến toàn cục
model = None
device = None

# Hàm khởi tạo mô hình YOLOv5
def initialize_model():
    global model, device
    model_path = r'D:\App\DLproject\yolov5\runs\train\exp\weights\best.pt'

    if not os.path.exists(model_path):
        logging.error(f"Không tìm thấy tệp mô hình tại: {model_path}")
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {model_path}")

    # Kiểm tra và sử dụng GPU nếu khả dụng
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sử dụng torch.hub.load với device được chỉ định
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, device=str(device))
    
    model.eval()
    
    logging.info("Các lớp của mô hình: %s", model.names)
    logging.info(f"Sử dụng thiết bị: {device}")
    
    return model, device

# Hàm nhận dạng văn bản từ biển số xe
def recognize_text_from_plate(plate_img):
    try:
        reader = easyocr.Reader(['en', 'vi'])  
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        result = reader.readtext(binary_img, detail=1)
        if result:
            text, confidence = result[0][-2], result[0][-1]
            if confidence > 0.1:  # Tăng độ tin cậy
                return text.strip()
        return ""
    except Exception as e:
        logging.error(f"Lỗi trong quá trình OCR với EasyOCR: {e}")
        return ""

# Hàm ghi vào cơ sở dữ liệu
def log_entry(plate):
    try:
        conn = sqlite3.connect('vehicle_log.db')
        cursor = conn.cursor()
        
        # Kiểm tra xe đã vào trong 5 phút gần đây chưa
        cursor.execute("""
            SELECT COUNT(*) FROM VehicleLogs 
            WHERE plate = ? AND gate = 'entry' AND entry_time >= datetime('now', '-5 minutes')
        """, (plate,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            cursor.execute("""
                INSERT INTO VehicleLogs (plate, entry_time, gate) 
                VALUES (?, datetime('now'), 'entry')
            """, (plate,))
            conn.commit()
            logging.info(f"Đã ghi nhận xe vào: {plate}")
            
            # Gửi thông tin qua Socket.IO
            socketio.emit('vehicle_log_update', {
                'plate': plate,
                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': None
            })
        
        conn.close()
    except Exception as e:
        logging.error(f"Lỗi trong quá trình ghi log entry: {e}")

def log_exit(plate):
    try:
        conn = sqlite3.connect('vehicle_log.db')
        cursor = conn.cursor()
        
        # Kiểm tra xe đã ra trong 5 phút gần đây chưa
        cursor.execute("""
            SELECT COUNT(*) FROM VehicleLogs 
            WHERE plate = ? AND gate = 'exit' AND exit_time >= datetime('now', '-5 minutes')
        """, (plate,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            cursor.execute("""
                INSERT INTO VehicleLogs (plate, exit_time, gate) 
                VALUES (?, datetime('now'), 'exit')
            """, (plate,))
            conn.commit()
            logging.info(f"Đã ghi nhận xe ra: {plate}")
            
            # Gửi thông tin qua Socket.IO
            socketio.emit('vehicle_log_update', {
                'plate': plate,
                'entry_time': None,
                'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        conn.close()
    except Exception as e:
        logging.error(f"Lỗi trong quá trình ghi log exit: {e}")

# Route cho dashboard
@app.route('/')
def dashboard():
    return render_template('dashboard1.html')

# Route cho video feed
@app.route('/video_feed')
def video_feed():
    return Response(process_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Hàm phát hiện đối tượng
def detect_objects(frame):
    global model, device
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    detections = results.xyxy[0].cpu().numpy()
    
    cars = []
    plates = []
    for *box, conf, cls in detections:
        cls = int(cls)
        if cls == 2 and conf > 0.9:  # Phát hiện xe   
            xmin, ymin, xmax, ymax = map(int, box)
            car_img = frame[ymin:ymax, xmin:xmax]
            cars.append({'bbox': (xmin, ymin, xmax, ymax), 'img': car_img, 'conf': conf})
        elif cls == 80 and conf > 0.9:  # Phát hiện biển số
            xmin, ymin, xmax, ymax = map(int, box)
            plate_img = frame[ymin:ymax, xmin:xmax]
            plates.append({'bbox': (xmin, ymin, xmax, ymax), 'img': plate_img, 'conf': conf})
        
    return cars, plates

# Hàm xử lý video stream thứ nhất
def process_video_stream():
    global model, device
    video_source = r"D:\App\DLproject\cong_vao.mp4"
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logging.error(f"Không thể mở video nguồn tại: {video_source}")
        return

    frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.warning("Không thể đọc khung hình từ video.")
            break

        frame_count += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            cars, plates = detect_objects(frame)
        except Exception as e:
            logging.error(f"Lỗi trong detect_objects: {e}")
            continue

        # Bounding box xe
        for car in cars:
            xmin, ymin, xmax, ymax = car['bbox']
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame.shape[1], xmax)
            ymax = min(frame.shape[0], ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
            cv2.putText(frame, "Car", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 0, 255), 3)

        # Bounding box biển số
        for plate in plates:
            xmin, ymin, xmax, ymax = plate['bbox']
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame.shape[1], xmax)
            ymax = min(frame.shape[0], ymax)
            
            plate_text = recognize_text_from_plate(plate['img'])
            logging.info(f"Biển số nhận diện được: {plate_text}")
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
            cv2.putText(frame, "Plate", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 3)
            
            if plate_text:
                log_entry(plate_text)

        # Hiển thị thống kê
        status_text = f"Cars: {len(cars)} | Plates: {len(plates)}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), font,
                    font_scale, (255, 255, 255), thickness)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Không thể mã hóa khung hình.")
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Hàm xử lý video stream thứ hai
def process_video_stream_2():
    global model, device
    video_source = r"D:\App\DLproject\cong_ra.mp4"
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logging.error(f"Không thể mở video nguồn tại: {video_source}")
        return

    frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.warning("Không thể đọc khung hình từ video.")
            break

        frame_count += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            cars, plates = detect_objects(frame)
        except Exception as e:
            logging.error(f"Lỗi trong detect_objects: {e}")
            continue

        for car in cars:
            xmin, ymin, xmax, ymax = car['bbox']
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame.shape[1], xmax)
            ymax = min(frame.shape[0], ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
            cv2.putText(frame, "Car", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 0, 255), 3)

        for plate in plates:
            xmin, ymin, xmax, ymax = plate['bbox']
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame.shape[1], xmax)
            ymax = min(frame.shape[0], ymax)
            
            plate_text = recognize_text_from_plate(plate['img'])
            logging.info(f"Biển số nhận diện được: {plate_text}")
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
            cv2.putText(frame, "Plate", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 3)
            
            if plate_text:
                log_exit(plate_text)

        status_text = f"Cars: {len(cars)} | Plates: {len(plates)}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), font,
                    font_scale, (255, 255, 255), thickness)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Không thể mã hóa khung hình.")
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route cho video feed thứ hai
@app.route('/video_feed_2')
def video_feed_2():
    return Response(process_video_stream_2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/supervise')
def supervise():
    return render_template('supervise.html')

# Hàm main
if __name__ == "__main__":
    init_db()
    try:
        model, device = initialize_model()
    except Exception as e:
        logging.critical(f"Không thể khởi tạo mô hình: {e}")
        exit(1)

    flask_thread = threading.Thread(target=socketio.run, args=(app,))
    flask_thread.daemon = True
    flask_thread.start()

    # Khởi tạo video thread thứ hai
    video_thread_2 = threading.Thread(target=process_video_stream_2)
    video_thread_2.daemon = True
    video_thread_2.start()
    flask_thread.join()
