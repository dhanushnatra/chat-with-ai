import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from threading import Thread
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import speech_recognition as sr
from gtts import gTTS
import pygame
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import logging
import asyncio
import webbrowser
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Initialize Flask App
app = Flask(__name__, static_folder='static')
CORS(app)


# --- Document Processing Module ---
class DocumentProcessor:
    """Handles loading and preprocessing of documents."""

    def __init__(self, max_chunk_length=300):
        self.max_chunk_length = max_chunk_length

    def load_document(self, filepath):
        """Loads a single document from the given filepath."""
        filename = os.path.basename(filepath)
        content = ""
        if filepath.endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logging.error(f"Error reading txt file '{filename}': {e}")
                return None, None

        elif filepath.endswith('.pdf'):
            try:
                reader = PdfReader(filepath)
                content = '\n'.join(page.extract_text() for page in reader.pages)
            except Exception as e:
                logging.error(f"Error reading pdf file '{filename}': {e}")
                return None, None
        else:
             logging.warning(f"Unsupported file format: '{filename}'")
             return None, None

        return filename, content

    def chunk_document(self, filename, content):
            """Splits document content into smaller, more manageable chunks."""
            chunks = []
            sources = []
            if content:
                for chunk in content.split("\n\n"):
                    chunk = chunk.strip()
                    if chunk and len(chunk) > self.max_chunk_length:
                        sentences = chunk.split('. ')
                        chunk = '. '.join(sentences[:min(len(sentences), 3)]) + '...'
                    if chunk:
                         chunks.append(chunk)
                         sources.append(filename)
            return chunks, sources

# --- Embedding and Retrieval Module ---
class EmbeddingManager:
    """Manages embeddings for text chunks and performs similarity searches."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logging.info("Loading the embedding model...")
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.chunks = []
        self.chunk_sources = []

    def generate_embeddings(self, chunks, sources):
        """Generates embeddings for the provided text chunks."""
        # Backend (app.py - Continued)
        logging.info("Generating embeddings for document chunks...")
        self.chunk_embeddings = self.model.encode(chunks, normalize_embeddings=True)
        self.chunks = chunks
        self.chunk_sources = sources
        logging.info("Embeddings generation complete!")

    def query(self, question, threshold=0.4):
            """Finds the most relevant chunk for a given question based on cosine similarity."""
            if self.chunk_embeddings is None:
                logging.warning("No embeddings found , please load documents to generate embeddings")
                return None

            # Embed the query
            query_embedding = self.model.encode(question, normalize_embeddings=True)

            # Compute cosine similarity with chunks
            similarities = util.cos_sim(query_embedding, self.chunk_embeddings)

            # Find the best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[0][best_match_idx].item()

            # Check if similarity exceeds threshold
            if best_similarity >= threshold:
                return {
                    'source_document': self.chunk_sources[best_match_idx],
                    'similarity': best_similarity,
                    'content': self.chunks[best_match_idx]
                }
            else:
               logging.info(f"No relevant information found, best similarity was {best_similarity:.2f}, the threshold was {threshold}")
               return None

# --- Speech Processing Module ---
class SpeechProcessor:
    """Handles speech recognition and text-to-speech functionality."""

    def __init__(self, language='en-gb', tld='co.uk'):
        self.language = language
        self.tld = tld
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False

    def start_listening(self):
         """Start listening for audio from the microphone"""
         self.is_listening = True
         logging.info("Starting microphone listening")
    def stop_listening(self):
        """Stop listening for audio from the microphone"""
        self.is_listening = False
        logging.info("Stopped microphone listening")

    def recognize_speech_from_microphone(self):
            """Recognizes speech from the microphone using Google Speech Recognition."""
            if not self.is_listening:
               logging.warning("Microphone is not active")
               return None

            try:
                with self.microphone as source:
                    logging.info("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source, phrase_time_limit = 10)
            except Exception as e:
                 logging.error(f"Error accessing microphone: {e}")
                 return None

            try:
                logging.info("Recognizing...")
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                logging.warning("Sorry, I couldn't understand that.")
                return None
            except sr.RequestError as e:
                 logging.error(f"Could not request results from Google Speech Recognition service; {e}")
                 return None
            except Exception as e:
                logging.error(f"General error during speech recognition {e}")
                return None

    def text_to_speech(self, text, output_file="response.mp3"):
        """Converts text to speech using gTTS and plays it using pygame."""
        try:
             tts = gTTS(text=text, lang=self.language, tld=self.tld, slow=False)
             tts.save(output_file)
             pygame.mixer.init()
             pygame.mixer.music.load(output_file)
             pygame.mixer.music.play()
             while pygame.mixer.music.get_busy():
                continue
             pygame.mixer.quit()
             os.remove(output_file)
        except Exception as e:
            logging.error(f"Error during text-to-speech: {e}")

# --- Chat Handling Module ---
class ChatHandler:
    """Handles different chat functionalities with the LLM."""

    def __init__(self, llm_model="llama3.2", streaming=True):
        self.llm = OllamaLLM(model=llm_model, streaming=streaming)
        self.history = []  # Initialize an empty chat history

    def handle_conversation(self, prompt_template, user_input, context=""):
        """Manages a general conversational AI interaction with the LLM."""
        try:
           formatted_history = "\n".join(f"{turn['speaker']}: {turn['message']}" for turn in self.history)
           self.history.append({"speaker":"User", "message":user_input}) #Add user input to history
           result = ""
           for chunk in (prompt_template | self.llm).stream({"context": context, "question": user_input, "history": formatted_history}):
                 result += chunk
           self.history.append({"speaker":"Max", "message":result})  # Add LLM's response to history
           return result
        except Exception as e:
            logging.error(f"Error during conversation: {e}")
            return None

    def handle_rag_with_ai(self, rag, prompt_template, question):
            """Manages document-based Q&A with AI using the LLM."""

            result = rag.query(question)
            if result:
                document_context = result['content']
                logging.info(f"Source Document: {result['source_document']}")
                logging.info(f"Relevance: {result['similarity']:.2f}")
                ai_input = {"context": document_context, "question": question}
                ai_answer = ""
                for chunk in (prompt_template | self.llm).stream(ai_input):
                     ai_answer += chunk
                return ai_answer, result['source_document']
            else:
                return None,None

    def handle_voice_chat(self, prompt_template, speech_processor, user_input, context=""):
            """Manages voice-based conversation with the LLM."""
            try:
                formatted_history = "\n".join(f"{turn['speaker']}: {turn['message']}" for turn in self.history)
                self.history.append({"speaker":"User", "message":user_input}) # add user input to history
                result = ""
                for chunk in (prompt_template | self.llm).stream({"context": context, "question": user_input, "history": formatted_history}):
                   result += chunk
                self.history.append({"speaker":"Max", "message":result}) # add llm response to history
                return result
            except Exception as e:
                 logging.error(f"Error during voice chat: {e}")
                 return None

# --- Flask Routes ---
# Initialize needed classes
doc_processor = DocumentProcessor()
embed_manager = EmbeddingManager()
speech_processor = SpeechProcessor()
chat_handler = ChatHandler()
conversational_prompt = ChatPromptTemplate.from_template("""
You are Max, a friendly and engaging assistant. Respond conversationally with short, correct, and fun answers while maintaining a helpful tone and acting like a enthusiastic person. Always keeps you responses short unless the user requests for a explaintion or a talking about a specifc topic . You will try to undersatnd the emotions of the user chating with you. always understand the context before answering
Conversation history:
{history}
Context: {context}
User: {question}
Max: 

    """)

document_prompt = ChatPromptTemplate.from_template("""
    Use the following document context to answer the question concisely and informatively.

    Context: {context}
    Question: {question}
    Answer:
    """)

voice_prompt = ChatPromptTemplate.from_template("""
        Your name is Max, a voice conversational AI with a human-like personality and wit . Respond in a very short and engaging way. talk like a human , make jokes and puns to make it interesting . taking the history and context into consideration before speaking .
        Conversation history:
        {history}
        Current question:
        {question}
        Max:
    """)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the uploading of a document and returns the response to the user"""
    if 'file' not in request.files:
         return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Save the file to a temporary location
        temp_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        filename, content = doc_processor.load_document(temp_path)
        if filename and content:
            chunks, sources = doc_processor.chunk_document(filename, content)
            embed_manager.generate_embeddings(chunks,sources)
            return jsonify({'success': 'File uploaded successfully', 'filename':filename}), 200
        else:
             return jsonify({'error': 'Error loading document or invalid document'}), 400

    except Exception as e:
        logging.error(f"Error processing upload : {e}")
        return jsonify({'error': f'Error processing upload: {e}'}), 500
    finally:
        if os.path.exists(temp_path):
           os.remove(temp_path)


@app.route('/rag_query', methods=['POST'])
def rag_query():
    """Handles RAG queries against the documents that have been uploaded"""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    question = data['question']

    ai_answer, document_name = chat_handler.handle_rag_with_ai(embed_manager, document_prompt, question)
    if ai_answer:
        return jsonify({'response': ai_answer, 'document_name':document_name}), 200
    else:
      return jsonify({'response': 'No relevant information found.'}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """ Handles the general chat using the llm model"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    message = data['message']
    response = chat_handler.handle_conversation(conversational_prompt, message)
    if response:
        return jsonify({'response': response}), 200
    else:
        return jsonify({'error': 'Error during chat'}), 500

@app.route('/start_voice_chat', methods=['POST'])
def start_voice_chat():
    """Starts voice chat by activating the microphone listener"""
    speech_processor.start_listening()
    return jsonify({"message":"Voice chat started"})

@app.route('/voice_chat_stream', methods=['POST'])
def voice_chat_stream():
    """Handles the streaming of voice input, processing and generation"""
    if not speech_processor.is_listening:
        return jsonify({"error":"voice chat is not active "}),400
    try:
      user_input = speech_processor.recognize_speech_from_microphone()
      if not user_input:
          logging.warning("No speech detected from microphone, continuing to listen")
          return jsonify({'response':""}),200

      response = chat_handler.handle_voice_chat(voice_prompt, speech_processor, user_input)
      if response:
          speech_processor.text_to_speech(response)
          return jsonify({'response': user_input}), 200  # Return user input for display
      else:
          return jsonify({'error': 'Error during voice chat'}), 500
    except Exception as e:
         logging.error(f"An unexpected error occured {e}")
         return jsonify({"error": f"An unexpected error occured {e}"}), 500


@app.route('/stop_voice_chat', methods=['POST'])
def stop_voice_chat():
    """Stops the ongoing voice chat and disables the microphone"""
    speech_processor.stop_listening()
    return jsonify({"message":"Voice chat stopped"})


@app.route('/', methods=['GET'])
def serve_static_files():
    """Serves the static files for the web application."""
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
        
    