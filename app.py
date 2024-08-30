from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import sqlite3
import dashscope
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from http import HTTPStatus
import logging
import fitz
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import io


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
dashscope.api_key = "sk-447f61df00ca4dcd84feeed099b3e630"

ELASTICSEARCH_HOST = "http://es-sg-ju33w74pb0002nt56.public.elasticsearch.aliyuncs.com:9200"  
ELASTICSEARCH_USERNAME = "elastic"  
ELASTICSEARCH_PASSWORD = "100%Winrate"

elvenlab_client = ElevenLabs(
    api_key="sk_96e9eae874414e7c81d4cfc35841233b5505b99180972bc7",
)

es = Elasticsearch(
    [ELASTICSEARCH_HOST],
    basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
)


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

        

        if "search" in user_message.lower():  
            search_term = user_message.lower().replace("search", "").strip()
            
            search_response = es.search(index="pdf-files", body={
                "query": {
                    "match": {
                        "filename": search_term
                    }
                }
            })
            print("Search response:", search_response)

            if 'hits' in search_response and 'hits' in search_response['hits']:
                documents = [
                    {
                        "id": hit["_id"],
                        "filename": hit["_source"].get("filename"),
                        "content": hit["_source"].get("content")
                    }
                    for hit in search_response['hits']['hits']
                ]
            else:
                documents = []
            

            search_results_text = "Relevant documents found:\n" + "\n".join(
                [f"Filename: {doc['filename']}, Content snippet: {doc['content'][:100]}" for doc in documents]
            )
            formatted_messages.append({"role": "system", "content": search_results_text})

        formatted_messages.append({"role": "user", "content": user_message})

        print("Formatted messages:", formatted_messages)

        # Call QWEN
        response = dashscope.Generation.call(
            "qwen-max",
            messages=formatted_messages,
            result_format='message',
            stream=False  
        )

        if response.status_code == HTTPStatus.OK:
            bot_message = response.output.choices[0]['message']['content']

            
            cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ('user', user_message))
            cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ('bot', bot_message))
            conn.commit()

            return jsonify({"botMessage": bot_message})
        else:
            print(f'Request id: {response.request_id}, Status code: {response.status_code}, '
                  f'error code: {response.code}, error message: {response.message}')
            return jsonify({"error": "An error occurred with the AI service. Please try again later."}), 500

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Error."}), 500

    finally:
        if conn:
            conn.close()

@app.route('/api/tts', methods=['POST'])
def tts():
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        voice_settings = VoiceSettings(
            stability=0.2,  
            similarity_boost=0.5,  
            style=0.75
        )

        response_generator = elvenlab_client.text_to_speech.convert(
            voice_id="OKanSStS6li6xyU1WdXa",  
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            voice_settings=voice_settings
        )

        audio_data = io.BytesIO()
        for chunk in response_generator:
            audio_data.write(chunk)
        
        audio_data.seek(0)

        return send_file(
            audio_data,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='output.mp3'
        )

    except Exception as e:
        logger.error(f"Error generating TTS: {str(e)}")
        return jsonify({'error': 'An error occurred while processing text-to-speech.'}), 500
    

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if not es:
        return jsonify({"error": "Elasticsearch is not available"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""

            for page in doc:
                text += page.get_text()

            try:
                response = es.index(index="pdf-files", document={"content": text, "filename": file.filename})
                response_dict = {
                    "result": response["result"],
                    "_id": response["_id"],
                    "_index": response["_index"],
                    "_version": response["_version"]
                }
                return jsonify({"message": "File uploaded and indexed successfully.", "es_response": response_dict})
            except es_exceptions.RequestError as e:
                logger.error(f"Elasticsearch indexing error: {str(e)}")
                return jsonify({"error": "Failed to index the file in Elasticsearch."}), 500

        except fitz.FileDataError as e:
            logger.error(f"PDF processing error: {str(e)}")
            return jsonify({"error": "Failed to process the PDF file. The file might be corrupt."}), 400
        except Exception as e:
            logger.error(f"Unexpected error while processing the file: {str(e)}")
            return jsonify({"error": "An unexpected error occurred while processing the file."}), 500

    return jsonify({"error": "Invalid file type. Only PDF is allowed."}), 400


if __name__ == '__main__':
    app.run(port=5500, debug=True)