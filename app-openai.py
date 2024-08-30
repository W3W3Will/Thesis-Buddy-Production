from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
from openai import OpenAI

client = OpenAI(
            api_key = "sk-B2xWy1VfL3FpyCXos-KEp8djIcdKWS9FEWlvEB1AKQT3BlbkFJP-G_OSFTENxpa7zcWmVoeaRx22BiAqT-JcNzDRUTgA"
)

app = Flask(__name__)
CORS(app)

def connect_db():
    return sqlite3.connect('chat.db')

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/api/chat', methods=['POST'])
def chat():
    if request.form:
        user_message = request.form.get('message')
    else:
        user_message = request.data.decode('utf-8')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM (SELECT * FROM messages ORDER BY id DESC LIMIT 10) ORDER BY id ASC")
        messages = cursor.fetchall()
        
        formatted_messages = []
        for msg in messages:
            role = "assistant" if msg[1] == "bot" else msg[1]
            formatted_messages.append({"role": role, "content": msg[2]})

        formatted_messages.append({"role": "user", "content": user_message})

        print("Formatted messages:", formatted_messages)

        chat_completion = client.chat.completions.create(
            messages=formatted_messages,
            model="gpt-4-turbo"
        )

        bot_message = chat_completion.choices[0].message.content

        cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ('user', user_message))
        cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ('bot', bot_message))
        conn.commit()

        return jsonify({"botMessage": bot_message})


    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "An error occurred with the AI service. Please try again later."}), 500

    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    app.run(port=5500, debug=True)
