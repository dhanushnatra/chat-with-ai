from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
from responder import askAI

app = Flask(__name__)

CORS(app)

@app.route('/chat', methods=['POST'])
def respond():
    data = request.get_json()
    print(data)
    prompt = data['message']
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        response:str = askAI(prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def home():
    return send_from_directory("static", "index.html")

if __name__ == '__main__':
    app.run(debug=True, port=5000,host="0.0.0.0")