# servers.py - Run all servers in one application on Python 3.12

import socket
import json
import threading
import time
import cv2
import numpy
import face_recognition
import sys
import ast
import os
from dotenv import load_dotenv, dotenv_values

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# LTM Fact Extraction

def filter_candidates(candidates):
    # Remove any conversation pairs where the assistant's response indicates uncertainty
    filtered = []
    uncertain_phrases = ["unknown", "i don't know", "not sure", "uncertain", "haven't"]
    for input, reply in candidates:
        reply_lower = reply.lower()
        if any(phrase in reply_lower for phrase in uncertain_phrases):
            continue
        filtered.append((input, reply))
    return filtered

def extract_facts_from_ltm_generic(ltm_entries):
    # Format filtered LTM pairs into a summary string
    filtered_candidates = filter_candidates(ltm_entries)
    if not filtered_candidates:
        return ""
    combined_text = " ".join(["User said: '{}' and Pepper replied: '{}'.".format(input, reply) for input, reply in filtered_candidates])
    return combined_text

# Long Term Memory (Milvus)

# Initialise the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.getenv("MODEL_CACHE_DIR"))
milvus_uri = os.getenv("MILVUS_IP")
milvus_token = os.getenv("MILVUS_TOKEN")

# Milvus client and collection setup
client = MilvusClient(uri=milvus_uri, token=milvus_token)
collection_name_memory = "chat_memory"

# Drop old collections
# for coll in ["face_embeddings", "chat_memory"]:
#     if coll in client.list_collections():
#         client.drop_collection(coll)
#         print("Dropped collection:", coll)
#     else:
#         print("Collection", coll, "does not exist.")

if collection_name_memory not in client.list_collections():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_fields=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("user", DataType.VARCHAR, max_length=255)
    schema.add_field("user_input", DataType.VARCHAR, max_length=1024)
    schema.add_field("bot_response", DataType.VARCHAR, max_length=1024)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384) # For MiniLM model

    index_params = client.prepare_index_params()
    index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="L2")

    client.create_collection(collection_name=collection_name_memory, schema=schema, index_params=index_params)
    print("[LTM] Created chat memory collection.")
else:
    print("[LTM] Collection already exists.")

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def store_conversation(user, user_input, bot_response):
    # Generate an embedding for the entire interaction
    embedding = get_embedding(user_input + " " + bot_response)
    data = [{
        "user": user,
        "user_input": user_input,
        "bot_response": bot_response,
        "embedding": embedding
    }]
    client.insert(collection_name=collection_name_memory, data=data)

def retrieve_similar_conversations(query, user, top_k=10):
    # Search Milvus for past interactions similar to query
    query_embedding = get_embedding(query)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = client.search(
        collection_name=collection_name_memory,
        data=[query_embedding],
        anns_field="embedding",
        search_params=search_params,
        limit=top_k,
        output_fields=["user", "user_input", "bot_response"]
    )
    filtered = []
    if results and results[0]:
        for res in results[0]:
            # Only include conversations that are tagged with the current user
            stored_user = res["entity"].get("user")
            if stored_user == user:
                filtered.append((res["entity"]["user_input"], res["entity"]["bot_response"]))
    return filtered

# Short Term Memory

class ShortTermMemory:
    def __init__(self, max_size=10, filename="short_term_memory.json"):
        self.memory = []
        self.max_size = max_size
        self.filename = filename
        self.load_memory()

    def add_interaction(self, user_input, bot_response):
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": bot_response})
        if len(self.memory) > self.max_size * 2:
            self.memory.pop(0)
            self.memory.pop(0)
        self.save_memory()

    def get_context(self):
        return self.memory

    def save_memory(self):
        with open(self.filename, "w") as f:
            json.dump(self.memory, f, indent=4)

    def load_memory(self):
        try:
            with open(self.filename, "r") as f:
                self.memory = json.load(f)
        except (IOError, ValueError):
            self.memory = []

user_contexts = {}
default_stm = ShortTermMemory(max_size=5, filename="stm_default.json")

# GroqAI Server

# Initialise the Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        os.getenv("SYSTEM_PROMPT")
    )
}

def groq_handle_client(client_socket, address):
    print(f"[GroqAI] Connection from: {address}")
    with client_socket:
        while True:
            try:
                data = client_socket.recv(4096)
                if not data:
                    break

                message = json.loads(data.decode("utf-8"))
                user_input = message.get("content", "")
                reset_flag = message.get("reset", False)
                user = message.get("user", None)
                emotion = message.get("emotion", None)
                if user:
                    if reset_flag or user not in user_contexts:
                        user_contexts[user] = ShortTermMemory(max_size=5, filename="stm_{}.json".format(user))
                    stm = user_contexts[user]
                    if reset_flag:
                        stm.memory = []  # Clear previous context
                else:
                    stm = default_stm
                    user = "default"

                # If reset flag was sent, acknowledge and skip processing
                if reset_flag:
                    client_socket.sendall(json.dumps({"content": "Context reset."}).encode("utf-8"))
                    continue

                full_context = [SYSTEM_PROMPT]
                full_context.append({"role": "system", "content": "The current user is " + user})
                if emotion:
                    full_context.append({"role": "system", "content": "The current user's emotional state is " + emotion})
                recent_context = stm.get_context()
                
                # Always retrieve similar conversations from LTM
                ltm_entries = retrieve_similar_conversations(user_input, user, top_k=10)

                fact_summary = extract_facts_from_ltm_generic(ltm_entries)
                if fact_summary:
                    fact_system_note = {"role": "system", "content": "Long-term memory facts: " + fact_summary}
                    full_context.append(fact_system_note)
                
                # If STM is empty, add a system note containing long term memory facts
                if not recent_context and ltm_entries:
                    facts = " ".join(["User said: '{}' and Pepper replied: '{}'.".format(inp, rep)
                                      for inp, rep in ltm_entries])
                    system_note = {"role": "system", "content": "Long-term memory facts: " + facts}
                    full_context.append(system_note)
                else:
                    full_context.extend(recent_context)

                    for inp, rep in ltm_entries:
                        full_context.append({"role": "user", "content": inp})
                        full_context.append({"role": "assistant", "content": rep})
                
                full_context.append({"role": "user", "content": user_input})
                
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=full_context,
                    temperature=1.2
                )
                bot_response = response.choices[0].message.content
                MAX_RESPONSE_LENGTH = 1000
                if len(bot_response) > MAX_RESPONSE_LENGTH:
                    print("[GroqAI DEBUG] Response too long. Truncating.")
                    bot_response = bot_response[:MAX_RESPONSE_LENGTH] + "...\n (Response shortened)"

                stm.add_interaction(user_input, bot_response)
                store_conversation(user, user_input, bot_response)
                print(f"[GroqAI] Assistant to {user}: {bot_response}")
                resp = {"content": bot_response}
                client_socket.sendall(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                print("[GroqAI] Error handling client:", e)
                break

def start_groq_server():
    SERVER_IP = os.getenv("SERVER_IP")
    PORT = os.getenv("SERVER_PORT")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, PORT))
    server_socket.listen(5)
    print("[GroqAI] Server listening on port", PORT)
    while True:
        client_sock, addr = server_socket.accept()
        threading.Thread(target=groq_handle_client, args=(client_sock, addr)).start()

# Face Server
collection_name_face = "face_embeddings"
SIMILARITY_THRESHOLD = 0.95

# Ensure collection exists
if collection_name_face not in client.list_collections():
    schema_face = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema_face.add_field("id", DataType.INT64, is_primary=True)
    schema_face.add_field("name", DataType.VARCHAR, max_length=255)
    schema_face.add_field("embedding", DataType.FLOAT_VECTOR, dim=128)

    index_params_face = client.prepare_index_params()
    index_params_face.add_index("embedding", index_type="AUTOINDEX")

    client.create_collection(collection_name=collection_name_face, schema=schema_face, index_params=index_params_face)
    print("[Face] Created face embedding collection.")
else:
    print("[Face] Collection already exists.")

def process_frame(frame_data):
    # Convert binary data into an OpenCV image
    frame_array = numpy.frombuffer(frame_data, dtype=numpy.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return frame

def normalise_embedding(embedding):
    norm = numpy.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return (embedding / norm).tolist()

def recognise_face(image):
    # Detect and match face
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    results = []
    
    for idx, (encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
        norm_encoding = normalise_embedding(encoding)
        
        search_result = client.search(
            collection_name=collection_name_face,
            data=[norm_encoding],
            limit=1,
            metric_type="IP", # Use inner product for normalised embeddings (equals cosine similarity)
            output_fields=["name"]
        )
        
        best_match = "Unknown"
        if search_result and search_result[0]:
            raw_item = search_result[0][0]
            
            # If the item is a string, attempt to parse it
            if isinstance(raw_item, str):
                try:
                    parsed_obj = ast.literal_eval(raw_item)
                    if isinstance(parsed_obj, list) and len(parsed_obj) > 0:
                        match_item = parsed_obj[0]
                    else:
                        match_item = {}
                except Exception as e:
                    match_item = {}
            else:
                match_item = raw_item
            
            score = match_item.get("distance")
            if score is not None:
                if score > SIMILARITY_THRESHOLD:
                    best_match = match_item["entity"].get("name", "Unknown")
        
        results.append({"name": best_match, "location": (top, right, bottom, left)})
    return results

def store_new_user(name, image):
    # Extract encoding and store into Milvus if not already present
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_image)
    if face_encodings:
        encoding = face_encodings[0]
        norm_encoding = normalise_embedding(encoding)
        
        # Check if this face is already registered
        search_result = client.search(
            collection_name=collection_name_face,
            data=[norm_encoding],
            limit=1,
            metric_type="IP",
            output_fields=["name"]
        )
        best_match = "Unknown"
        if search_result and search_result[0]:
            match = search_result[0][0]
            score = match.get("distance")
            if score is not None and score > SIMILARITY_THRESHOLD:
                print("Score:", score)
                best_match = match["entity"].get("name", "Unknown")
        if best_match != "Unknown":
            return "Already registered as " + best_match
        else:
            entity = [{"name": name, "embedding": norm_encoding}]
            client.insert(collection_name_face, data=entity)
            client.flush([collection_name_face])
            return "Stored " + name + " in database."
    return "Failed to store user."

def face_handle_client(client_socket, address):
    print(f"[Face] Connection from: {address}")
    try:
        header_data = b""
        while True:
            chunk = client_socket.recv(1024)
            if not chunk:
                break
            if b"\0" in chunk:
                header_data += chunk.split(b"\0")[0]
                break
            header_data += chunk
        header = json.loads(header_data.decode("utf-8"))

        # Read the rest of the frame data
        frame_data = b""
        bytes_to_receive = header.get("content_length", 0)
        while len(frame_data) < bytes_to_receive:
            frame_data += client_socket.recv(min(bytes_to_receive - len(frame_data), 4096))
        
        if header.get("type") == "frame":
            image = process_frame(frame_data)
            response = recognise_face(image)
            client_socket.sendall(json.dumps(response).encode("utf-8"))
        elif header.get("type") == "register":
            name = header.get("name")
            image = process_frame(frame_data)
            result = store_new_user(name, image)
            client_socket.sendall(result.encode("utf-8"))
        else:
            client_socket.sendall("Invalid request type".encode("utf-8"))
    except Exception as e:
        print("[Face] Error handling client:", e)
    finally:
        client_socket.close()

def start_face_server():
    SERVER_IP = os.getenv("SERVER_IP")
    PORT = os.getenv("FACE_PORT")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, PORT))
    server_socket.listen(5)
    print(f"[Face] Server listening on {SERVER_IP}:{PORT}")
    while True:
        client_sock, addr = server_socket.accept()
        threading.Thread(target=face_handle_client, args=(client_sock, addr)).start()

# -------------------- Main --------------------

def main():
    threading.Thread(target=start_groq_server, daemon=True).start() # Start GroqAI server thread
    threading.Thread(target=start_face_server, daemon=True).start() # Start face server thread

    print("Servers are running. Type 'exit' and press Enter to shut down.")
    while True:
        user_input = input()
        if user_input.strip().lower() == "exit":
            print("Exiting servers.")
            break
    sys.exit(0)

if __name__ == "__main__":
    main()
