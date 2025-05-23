# pepper.py - Pepper control for Python 2.7

from __future__ import print_function # Treat print as a statement
import qi
import time
import random
import numpy
import cv2
import socket
import json
import threading
import paramiko
import speech_recognition as sr
import wave
import os
import shutil
import re
from scp import SCPClient
from dotenv import load_dotenv, dotenv_values

# Global Configuration

load_dotenv()

active_user = "Unknown"
active_user_lock = threading.Lock()

# Network endpoints
PEPPER_IP = os.getenv("PEPPER_IP")
PEPPER_PORT = os.getenv("PEPPER_PORT")

GROQ_SERVER_IP = os.getenv("GROQ_SERVER_IP")
GROQ_SERVER_PORT = os.getenv("GROQ_SERVER_PORT")

FACE_SERVER_IP = os.getenv("FACE_SERVER_IP")
FACE_SERVER_PORT = os.getenv("FACE_SERVER_PORT")

# Audio thresholds
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 0.1

# Local recording directory
LOCAL_AUDIO_DIR = os.getenv("LOCAL_AUDIO_DIR")
# SSH/SCP for transfer
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=PEPPER_IP, username=os.getenv("PEPPER_USERNAME"), password=os.getenv("PEPPER_PASSWORD"))
scp = SCPClient(ssh.get_transport())

# Pepper Services
session = qi.Session()
session.connect("tcp://%s:%s" % (PEPPER_IP, PEPPER_PORT))

tts = session.service("ALTextToSpeech")
audio_service = session.service("ALAudioRecorder")
speech_service = session.service("ALSpeechRecognition")
camera_service = session.service("ALVideoDevice")
memory_service = session.service("ALMemory")
animation_player = session.service("ALAnimationPlayer")
posture_service = session.service("ALRobotPosture")
motion_service = session.service("ALMotion")
tracker_service = session.service("ALTracker")

# Animations

GREETING_ANIMATIONS = [
    "animations/Stand/Gestures/Hey_1",
    "animations/Stand/Gestures/Hey_3",
    "animations/Stand/Gestures/Hey_4",
    "animations/Stand/Gestures/Hey_6"
]

SPEAKING_ANIMATIONS = [
    "animations/Stand/BodyTalk/BodyTalk_1",
    "animations/Stand/BodyTalk/BodyTalk_2",
    "animations/Stand/BodyTalk/BodyTalk_3",
    "animations/Stand/BodyTalk/BodyTalk_4",
    "animations/Stand/BodyTalk/BodyTalk_5",
    "animations/Stand/BodyTalk/BodyTalk_6",
    "animations/Stand/BodyTalk/BodyTalk_7",
    "animations/Stand/BodyTalk/BodyTalk_8",
    "animations/Stand/BodyTalk/BodyTalk_9",
    "animations/Stand/BodyTalk/BodyTalk_10",
    "animations/Stand/BodyTalk/BodyTalk_11",
    "animations/Stand/BodyTalk/BodyTalk_12",
    "animations/Stand/BodyTalk/BodyTalk_13",
    "animations/Stand/BodyTalk/BodyTalk_14",
    "animations/Stand/BodyTalk/BodyTalk_15",
    "animations/Stand/BodyTalk/BodyTalk_16"
]

CONFUSED_ANIMATIONS = [
    "animations/Stand/Gestures/IDontKnow_1",
    "animations/Stand/Gestures/IDontKnow_2",
    "animations/Stand/Gestures/IDontKnow_3",
    "animations/Stand/Gestures/No_1",
    "animations/Stand/Gestures/No_2",
    "animations/Stand/Gestures/No_3",
    "animations/Stand/Gestures/No_8",
    "animations/Stand/Gestures/No_9"
]

THINKING_ANIMATIONS = [
    "animations/Stand/Gestures/Thinking_1",
    "animations/Stand/Gestures/Thinking_3",
    "animations/Stand/Gestures/Thinking_4",
    "animations/Stand/Gestures/Thinking_6",
    "animations/Stand/Gestures/Thinking_8",
    "animations/Stand/Waiting/Think_1",
    "animations/Stand/Waiting/Think_2",
    "animations/Stand/Waiting/Think_3"
]

def perform_animation(animation_list):
    def run_animation():
        animation = random.choice(animation_list)
        animation_player.run(animation, _async=True)
        posture_service.goToPosture("StandInit", 0.5) # Return to standing position after animation
    threading.Thread(target=run_animation).start()

# LED Indicators

leds_service = session.service("ALLeds")

def set_face_colour(r, g, b, fade_time):
    for side in ("Right", "Left"):
        leds_service.fade("%sFaceLedsRed" % side, r, fade_time)
        leds_service.fade("%sFaceLedsGreen" % side, g, fade_time)
        leds_service.fade("%sFaceLedsBlue" % side, b, fade_time)

def indicate_listening():
    # Solid green
    threading.Thread(target=set_face_colour, args=(0.0, 1.0, 0.0, 0.1)).start()

def indicate_thinking():
    # Solid blue
    threading.Thread(target=set_face_colour, args=(0.0, 0.0, 1.0, 0.1)).start()

def indicate_speaking():
    # Solid red
    threading.Thread(target=set_face_colour, args=(1.0, 0.0, 0.0, 0.1)).start()
    
# Conversation Context

def reset_conversation_context(user):
    # Attempt to reset the short-term memory for the user
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((GROQ_SERVER_IP, GROQ_SERVER_PORT))
        message = {"content": "", "user": user, "reset": True}
        client_socket.sendall(json.dumps(message))
        acknowledge = client_socket.recv(4096)
        print("Reset context acknowledgement:", acknowledge)
    except Exception as e:
        print("Error resetting conversation context:", e)
    finally:
        client_socket.close()

def send_registration_event(user):
    # Notify Groq that a new user has been registered
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((GROQ_SERVER_IP, GROQ_SERVER_PORT))
        message = {"content": "User registered as " + user, "user": user, "reset": False}
        client_socket.sendall(json.dumps(message))
        acknowledge = client_socket.recv(4096)
        print("Registration event acknowledgement:", acknowledge)
    except Exception as e:
        print("Error sending registration event:", e)
    finally:
        client_socket.close()

# Audio Handling

registration = False
ready_to_record = True

def download_audio(remote_path, path="speech_chunk.wav"):
    local_path = os.path.join(LOCAL_AUDIO_DIR, path)
    try:
        scp.get(remote_path, local_path)
        return local_path
    except Exception as e:
        print("Error downloading file:", e)
        return None

def is_speech_present(audio_file):
    try:
        wf = wave.open(audio_file, "rb")
        frames = wf.readframes(wf.getnframes())
        wf.close()
        audio_data = numpy.frombuffer(frames, dtype=numpy.int16)
        energy = numpy.sum(numpy.abs(audio_data)) / float(len(audio_data))
        return energy > SILENCE_THRESHOLD
    except Exception as e:
        print("Error processing audio file:", e)
        return False

def speech_to_text(audio_file):
    recogniser = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recogniser.record(source) # Removed adjust for ambient since it was causing issues
        recognised_text = recogniser.recognize_google(audio) #language="en-GB"
        return recognised_text
    except sr.UnknownValueError:
        print("Could not understand.")
        perform_animation(CONFUSED_ANIMATIONS)
        return None
    except sr.RequestError as e:
        print("Google Speech Recognition error:", e)
        return None

def append_audio(full_file, new_chunk):
    try:
        chunk_wav = wave.open(new_chunk, "rb")
        chunk_frames = chunk_wav.readframes(chunk_wav.getnframes())
        chunk_wav.close()
        if os.path.exists(full_file):
            full_wav = wave.open(full_file, "rb")
            full_params = full_wav.getparams()
            full_frames = full_wav.readframes(full_wav.getnframes())
            full_wav.close()
            combined_wav = wave.open(full_file, "wb")
            combined_wav.setparams(full_params)
            combined_wav.writeframes(full_frames + chunk_frames)
            combined_wav.close()
        else:
            shutil.move(new_chunk, full_file)
    except Exception as e:
        print("Error appending audio:", e)

def cleanup_audio_files():
    # Removes full_speech.wav and leftover chunk files
    try:
        full_file = os.path.join(LOCAL_AUDIO_DIR, "full_speech.wav")
        chunk_file = os.path.join(LOCAL_AUDIO_DIR, "speech_chunk.wav")
        if os.path.exists(full_file):
            os.remove(full_file)
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
    except Exception as e:
        print("Error deleting files:", e)

def record_audio():
    # Record successive chunks, detect end of speech and hand off concatenated file
    if registration:
         cleanup_audio_files()
         return

    full_audio_path = os.path.join(LOCAL_AUDIO_DIR, "full_speech.wav")
    if os.path.exists(full_audio_path):
        os.remove(full_audio_path)
        print("Deleted old full_speech.wav")

    audio_service.stopMicrophonesRecording()
    silence_start = None
    speech_detected = False
    thinking_triggered = False
    chunk_index = 0

    current_chunk_index = chunk_index % 10
    current_chunk_path = "/home/nao/speech_chunk_{}.wav".format(current_chunk_index)
    indicate_listening()
    audio_service.startMicrophonesRecording(current_chunk_path, "wav", 40000, (0, 0, 1, 0))

    while True:
        if registration:
            break

        time.sleep(0.5) # length of chunks

        audio_service.stopMicrophonesRecording()
        next_chunk_index = (chunk_index + 1) % 10
        next_chunk_path = "/home/nao/speech_chunk_{}.wav".format(next_chunk_index)
        audio_service.startMicrophonesRecording(next_chunk_path, "wav", 40000, (0, 0, 1, 0))

        local_chunk = download_audio(current_chunk_path, "speech_chunk_{}.wav".format(current_chunk_index))
        if not local_chunk:
            print("Failed to download chunk {}".format(current_chunk_index))
            chunk_index += 1
            current_chunk_path = next_chunk_path
            continue

        if is_speech_present(local_chunk):
            append_audio(full_audio_path, local_chunk)
            speech_detected = True
            silence_start = None
             if not thinking_triggered:
                 perform_animation(THINKING_ANIMATIONS)
                 thinking_triggered = True

        elif speech_detected:
            append_audio(full_audio_path, local_chunk)
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_DURATION:
                print("User finished speaking.")
                process_user_input(full_audio_path)
                cleanup_audio_files()
                break
        
        chunk_index += 1
        current_chunk_index = chunk_index % 10
        current_chunk_path = next_chunk_path

def record_short_audio(duration=2):
    # Records short utterance (for name prompts)
    global ready_to_record
    while not ready_to_record:
        time.sleep(0.05)

    short_audio_path = os.path.join(LOCAL_AUDIO_DIR, "short_speech.wav")
    if os.path.exists(short_audio_path):
        os.remove(short_audio_path)

    audio_service.stopMicrophonesRecording()
    chunk_path = "/home/nao/short_speech.wav"
    audio_service.startMicrophonesRecording(chunk_path, "wav", 40000, (0, 0, 1, 0))
    time.sleep(duration)
    audio_service.stopMicrophonesRecording()
    local_file = download_audio(chunk_path, "short_speech.wav")
    return local_file

# Name Registration

def get_user_name_via_voice():
    # Ask the user for their name
    global registration
    registration = True

    try:
        for attempt in range(3):
            indicate_speaking()
            tts.say("I do not recognise you. Please state your name.")
            indicate_listening()

            audio_file = record_short_audio(2)
            if audio_file:
                name_text = speech_to_text(audio_file)
                if name_text:
                    extracted = extract_name(name_text)
                    if extracted:
                        indicate_speaking()
                        tts.say("Nice to meet you, " + extracted + ".")
                        perform_animation(GREETING_ANIMATIONS)
                        return extracted
                    else:
                        indicate_speaking()
                        tts.say("I could not extract your name. Please try again.")
                else:
                    indicate_speaking()
                    tts.say("Please try again.")
            else:
                indicate_speaking()
                tts.say("I didn't catch that. Please try again.")
        indicate_speaking()
        tts.say("I did not recognise your name after several attempts.")
        return None
    finally:
        registration = False

def extract_name(text):
    # Identify name from potential string
    text_lower = text.lower()
    match = re.search(r"(?:my name is|i am|i'm)\s+(\w+)", text_lower)
    if match:
        return match.group(1).capitalize()
    else:
        words = text.split()
        return words[0].capitalize() if words else None

# User Input Processing

def process_user_input(audio_file):
    # Transcribes concatenated full_speech, attaches it to the active user, sends it to server with emotion, then speak the response
    global registration
    if registration:
         print("Registration in progress, ignoring conversational input.")
         cleanup_audio_files()
         return

    indicate_thinking()
    user_text = speech_to_text(audio_file)
    if not user_text:
         print("No clear speech detected.")
         perform_animation(CONFUSED_ANIMATIONS)
         return

    with active_user_lock:
         current_user = active_user

    detected_emotion = get_emotion()
    print("User:", user_text, "| Emotion:", detected_emotion)
    perform_animation(THINKING_ANIMATIONS)
    response = send_to_server(user_text, current_user, detected_emotion)
    print("Response:", response)
    indicate_speaking()
    tts.say(response)
    perform_animation(SPEAKING_ANIMATIONS)

def send_to_server(text, user, emotion=None):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((GROQ_SERVER_IP, GROQ_SERVER_PORT))
        message = {"content": text, "user": user}

        if emotion:
            message["emotion"] = emotion

        client_socket.sendall(json.dumps(message))
        response_data = client_socket.recv(4096)
        if not response_data:
            return "No response from server."

        response = json.loads(response_data)
        return response.get("content", "")
    except Exception as e:
        print("Connection error:", e)
        return "Connection error."
    finally:
        client_socket.close()

# Voice Thread

def voice_thread():
    # Continuous background loop for recording audio
    global ready_to_record
    while True:
        if ready_to_record:
            record_audio()
        time.sleep(0.1)

# Face Functions

def subscribe_camera(camera="camera_top", resolution=1, fps=30):
    # Subscribe to the specified camera and return stream
    global camera_link
    camera_index = 0 if camera == "camera_top" else 1
    colour_space = 13
    camera_link = camera_service.subscribeCamera("FaceStream" + str(random.random()), camera_index, resolution, colour_space, fps)
    return camera_link

def get_frame():
    # Retrieve latest image from subscribed camera
    image_raw = camera_service.getImageRemote(camera_link)
    if image_raw is None:
        return None
    image = numpy.frombuffer(image_raw[6], numpy.uint8).reshape(image_raw[1], image_raw[0], 3)
    return image

def send_frame_to_face_server(image):
    # Send JPEG-encoded frame to server
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((FACE_SERVER_IP, FACE_SERVER_PORT))
        ret, encoded_image = cv2.imencode(".jpg", image)
        if not ret:
            print("Failed to encode image.")
            return None
        frame_data = encoded_image.tobytes()
        header = {
            "type": "frame",
            "content_length": len(frame_data),
            "content_type": "binary"
        }
        header_bytes = json.dumps(header)
        client_socket.sendall(header_bytes + "\0")
        client_socket.sendall(frame_data)
        response = client_socket.recv(4096)
        client_socket.close()
        return json.loads(response)
    except Exception as e:
        print("Error sending frame:", e)
        return None

def register_new_user(image, provided_name=None):
    # Capture face image and register it with name
    global active_user
    if provided_name:
        new_user_name = provided_name
    else:
        new_user_name = get_user_name_via_voice()
        print("Registered user:", new_user_name)
    if new_user_name:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((FACE_SERVER_IP, FACE_SERVER_PORT))
            ret, encoded_image = cv2.imencode(".jpg", image)
            if not ret:
                print("Failed to encode image for registration.")
                return
            frame_data = encoded_image.tobytes()
            message = {"type": "register", "name": new_user_name, "content_length": len(frame_data)}
            header_bytes = json.dumps(message)
            client_socket.sendall(header_bytes + "\0")
            client_socket.sendall(frame_data)
            response = client_socket.recv(1024)
            client_socket.close()
            print("Registration response:", response)
            with active_user_lock:
                active_user = new_user_name
            reset_conversation_context(new_user_name)
            send_registration_event(new_user_name)
        except Exception as e:
            print("Error during registration:", e)
    else:
        print("User registration failed.")

def face_stream_thread():
    # Continuously send frames for recognition, handle user registration and switch active user
    global active_user
    subscribe_camera("camera_top", 1, 30)
    
    unknown_count = 0
    known_count = 0
    known_user = None
    FRAME_THRESHOLD = 5

    while True:
        frame = get_frame()
        if frame is None:
            continue

        face_results = send_frame_to_face_server(frame)
        if face_results and len(face_results) > 0:

            detected_name = face_results[0].get("name", "Unknown")          
            if detected_name == "Unknown":
                unknown_count += 1
                known_count = 0
            else:
                unknown_count = 0
                with active_user_lock:
                    if detected_name != active_user:
                        known_user = detected_name
                        known_count += 3
                    else:
                        known_count = 0
            # If unknown detections exceeds threshold, prompt for registration
            if unknown_count >= FRAME_THRESHOLD:
                new_name = get_user_name_via_voice()
                if new_name:
                    register_new_user(frame, provided_name=new_name)
                unknown_count = 0
            # If known detection user is different to current active user, update active user
            if known_count >= FRAME_THRESHOLD:
                if known_user and known_user != active_user:
                    tts.say("Hello again, " + known_user + ".")
                    perform_animation(GREETING_ANIMATIONS) # Can interfere with listening for Pepper
                    reset_conversation_context(known_user)
                    with active_user_lock:
                        active_user = known_user
                    known_count = 0
        else:
            print("No face detected in this frame.")
        time.sleep(0.1)

def get_emotion():
    # Retrieve the emotional state of the detected face from PeoplePerception
    emotions = ["neutral", "happy", "surprised", "angry", "sad"]
    try:
        people_list = memory_service.getData("PeoplePerception/PeopleList")
        if not people_list or len(people_list) == 0:
            return None

        person_id = people_list[0]
        properties = memory_service.getData("PeoplePerception/Person/" + str(person_id) + "/ExpressionProperties")
        if not properties:
            return None

        max_index = properties.index(max(properties))
        if properties[max_index] < 0.7:
            return None

        return emotions[max_index]
    except Exception as e:
        print("Error retrieving emotion:", e)
        return None

def start_face_tracking():
    try:
        motion_service.wakeUp()
        tracker_service.registerTarget("Face", 0.1)
        tracker_service.track("Face")
        print("Face tracking begun.")
        while True:
            time.sleep(1)
    except Exception as e:
        print("Tracking error:", e)

# Main

def main():
    session.service("ALAutonomousLife").setState("disabled") # Disable built-in autonomous movement

    # Subscribe to perception services
    session.service("ALPeoplePerception").subscribe("PeoplePerception") # Re-enable PeoplePerception - Emotions
    session.service("ALFaceCharacteristics").subscribe("PeoplePerception") # Re-enable Face Characteristics identification - Tracking


    threading.Thread(target=voice_thread).start()
    threading.Thread(target=face_stream_thread).start()
    threading.Thread(target=start_face_tracking).start()
    
    # Keep running
    while True:
        continue
        #time.sleep(1)

if __name__ == "__main__":
    main()

