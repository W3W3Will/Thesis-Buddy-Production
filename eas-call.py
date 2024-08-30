from flask import Flask, render_template, request, jsonify
import websocket
import json
import threading
import time
import queue

app = Flask(__name__)

EAS_WS_URL = "ws://testing.5388423794916646.ap-southeast-5.pai-eas.aliyuncs.com/"
EAS_AUTH_TOKEN = "MTUwMTFhMzIxOWIzZmI4ZmVlZjUwMTMzZjg3MGQ0NDc5ODA1MTVmNA=="

ws = None
ws_response_queue = queue.Queue()
ws_lock = threading.Lock()

def on_message(ws, message):
    if message == "":
        ws_response_queue.put("<EOS>")
    else:
        ws_response_queue.put(message)

def on_error(ws, error):
    ws_response_queue.put(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    time.sleep(5)
    create_ws_connection()

def on_open(ws):
    pass

def create_ws_connection():
    global ws
    ws = websocket.WebSocketApp(
        EAS_WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
        header=['Authorization: ' + EAS_AUTH_TOKEN]
    )
    wst = threading.Thread(target=ws.run_forever, kwargs={'ping_interval': 30, 'ping_timeout': 10})
    wst.daemon = True
    wst.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    
    params = {
        "prompt": user_message,
        "temperature": 0.9,
        "top_p": 0.1,
        "top_k": 30,
        "max_new_tokens": 2048,
        "do_sample": True
    }
    
    raw_req = json.dumps(params, ensure_ascii=False).encode('utf8')
    
    if not ws or not ws.sock or not ws.sock.connected:
        return jsonify({"botMessage": "WebSocket is not connected. Please try again later."})

    ws_response_queue.queue.clear()  
    ws.send(raw_req)
    
    last_message = ""
    timeout = time.time() + 30 
    
    while time.time() < timeout:
        try:
            message = ws_response_queue.get(timeout=1)
            if message == "<EOS>":
                break
            elif message.startswith("Error:"):
                return jsonify({"botMessage": message})
            last_message = message 
        except queue.Empty:
            continue
    
    if not last_message:
        return jsonify({"botMessage": "Sorry, I didn't receive a response in time. Please try again."})
    else:
        return jsonify({"botMessage": last_message.strip()})

if __name__ == '__main__':
    create_ws_connection()
    app.run(debug=True, port=5500)
