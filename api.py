from fastapi import FastAPI, File, UploadFile
import whisper
import os
import torch
import io
import logging
from pydub import AudioSegment
from datetime import datetime

# >----- Settings
DIRECTORY_PATH = 'audio'  # Provide path to temp audio directory
MODEL = "turbo"            # Choose model (you can replace with another one)
# --------------<

# Turn off extra logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the FastAPI app
app = FastAPI()

# Optimize memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the Whisper model
model = whisper.load_model(MODEL)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        service_duration = 0
        transcription_duration = 0
        
        # Save start time
        service_start_time = datetime.now()
    
        # Read the contents of the uploaded file
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        recording = AudioSegment.from_file(audio_file)

        # Create output directory if not exists
        os.makedirs(DIRECTORY_PATH, exist_ok=True)

        base, ext = os.path.splitext(file.filename)
        mp3_path = f"{DIRECTORY_PATH}/{base}.mp3"

        # Export the uploaded file to mp3
        recording.export(mp3_path, format='mp3')
        audio = AudioSegment.from_file(mp3_path, format="mp3")

        # Split the audio into fragments of 30 seconds each
        fragment_length = 30000  # 30 seconds in milliseconds
        fragments = [
            audio[i * fragment_length:(i + 1) * fragment_length]
            for i in range(len(audio) // fragment_length)
        ]
        # Handle the last fragment which might be shorter
        if len(audio) % fragment_length != 0:
            fragments.append(audio[len(audio) // fragment_length * fragment_length:])

        # Save fragments with sequential names
        fragment_filenames = []
        for i, fragment in enumerate(fragments):
            fragment_filename = f"{DIRECTORY_PATH}/fragment_{i:04d}.mp3"
            fragment.export(fragment_filename, format="mp3")
            fragment_filenames.append(fragment_filename)

        # Remove the temporary main audio file
        os.remove(mp3_path)
        
        service_end_time = datetime.now()
        service_duration += (service_end_time - service_start_time).total_seconds()

        # Perform transcription in order
        answer = ""
        transcription_start_time = datetime.now()
        for fragment_filename in sorted(fragment_filenames):
            logging.info(f"Processing file: {fragment_filename}")
            torch.cuda.empty_cache()  # Clear CUDA cache to avoid memory issues
            result = model.transcribe(audio=fragment_filename, language='ru')
            answer += result.get('text', "") + " "
            os.remove(fragment_filename)  # Remove fragment after transcription
           
        # Save end time    
        transcription_end_time = datetime.now()
        transcription_duration += (transcription_end_time - transcription_start_time).total_seconds()
        
        service_start_time = datetime.now()
        answer = answer.replace("  ", " ")
        words_count = len(answer.strip().split(" "))
        service_end_time = datetime.now()
        service_duration += (service_end_time - service_start_time).total_seconds()

        # Return the combined transcription result
        return {"transcribed": True, 
                "text": answer.strip(),
                "transcription_duration": transcription_duration,
                "words_count": words_count, 
                "service_duration": service_duration,
                "total_duration": transcription_duration + service_duration
                }

    # Handle exception
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return {"transcribed": False, "error": str(e)}
