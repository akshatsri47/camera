import os
import io
import cv2
import time
import threading
import queue
import uvicorn

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import google.generativeai as genai
import pyttsx3

# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyB2WefuRXPWAkxvhY5PapJIW5DoZFtNHsw")
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI()

# Global variables
latest_frame = None
audio_queue = queue.Queue()
description_interval = 10  # Check camera every 3 seconds as requested

# Initialize text-to-speech engine in its own thread
def tts_thread_function():
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Speed of speech
    
    while True:
        # Get text from queue
        text = audio_queue.get()
        if text == "STOP":  # Sentinel to stop the thread
            break
            
        try:
            print(f"Speaking: {text}")
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            
        # Mark the task as done
        audio_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_thread_function, daemon=True)
tts_thread.start()

# Function to speak text (adds to queue)
def speak_text(text):
    audio_queue.put(text)

# ------------------------
# Gemini Model Inference
# ------------------------
def run_gemini(query, image):
    message = f"""Analyze the image and answer the question below. 
    If the image is unrelated, ignore it and answer based on textual knowledge. 
    Keep your response very concise - no more than 1-2 sentences.\n\nQuestion: {query}"""
    
    try:
        response = gemini_model.generate_content([message, image])
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        return "Unable to analyze the image at this time."

# ------------------------
# Camera Thread
# ------------------------
def capture_camera_frames():
    global latest_frame
    last_description_time = 0
    last_description = ""
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("Camera opened successfully. Starting frame capture.")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.5)
                continue
                
            latest_frame = frame
            
            # Check if it's time for a new description
            current_time = time.time()
            if current_time - last_description_time >= description_interval:
                # Convert OpenCV frame (BGR) to PIL image (RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Get description from Gemini
                prompt = "Briefly describe what's visible in this camera feed."
                description = run_gemini(prompt, pil_img)
                
                # Only speak if description is different
                if description != last_description:
                    speak_text(description)
                    last_description = description
                
                last_description_time = current_time
                
            # Short sleep to reduce CPU usage
            time.sleep(0.1)
    except Exception as e:
        print(f"Camera thread error: {str(e)}")
    finally:
        cap.release()
        print("Camera released")

# ------------------------
# File Upload Endpoint
# ------------------------
@app.post("/process/")
async def process(query: str, image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        pil_img = Image.open(io.BytesIO(image_data))
        response = run_gemini(query, pil_img)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------
# Live Camera Describe Endpoint
# ------------------------
@app.get("/describe/")
def describe_from_camera():
    if latest_frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready yet")

    # Convert OpenCV frame (BGR) to PIL image (RGB)
    rgb_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    prompt = "Describe what can be seen in this image from a live camera feed."
    response = run_gemini(prompt, pil_img)
    
    # Also speak the description
    speak_text(response)

    return JSONResponse(content={"description": response})

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    # Start the camera thread
    camera_thread = threading.Thread(target=capture_camera_frames, daemon=True)
    camera_thread.start()
    
    # Give the camera time to initialize
    time.sleep(1)
    
    # Start the FastAPI server
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)