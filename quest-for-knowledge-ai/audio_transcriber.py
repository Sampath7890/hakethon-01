import speech_recognition as sr
import os
from datetime import datetime
import pydub
import tempfile
import wave
import sys
import subprocess
import platform
import json

class AudioTranscriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust recognition settings
        self.recognizer.energy_threshold = 300  # Lower threshold for better recognition
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause threshold
        self.recognizer.phrase_threshold = 0.3  # Lower phrase threshold
        self.recognizer.non_speaking_duration = 0.5  # Shorter non-speaking duration
        self.setup_chrome_flags()
        
    def setup_chrome_flags(self):
        """Set up Chrome flags for media stream handling"""
        try:
            # Check if running on Windows
            if platform.system() == 'Windows':
                # Get Chrome installation path
                chrome_paths = [
                    os.path.expandvars(r'%ProgramFiles%\Google\Chrome\Application\chrome.exe'),
                    os.path.expandvars(r'%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe'),
                    os.path.expandvars(r'%LocalAppData%\Google\Chrome\Application\chrome.exe')
                ]
                
                chrome_path = None
                for path in chrome_paths:
                    if os.path.exists(path):
                        chrome_path = path
                        break
                
                if chrome_path:
                    # Set Chrome flags
                    subprocess.run([
                        chrome_path,
                        '--use-fake-ui-for-media-stream',
                        '--disable-features=IsolateOrigins,site-per-process',
                        '--disable-site-isolation-trials'
                    ], check=True)
                    print("Chrome flags set successfully")
                else:
                    print("Warning: Chrome not found. Some features may not work properly.")
        except Exception as e:
            print(f"Warning: Could not set Chrome flags: {str(e)}")
            print("Some features may not work properly.")
        
    def convert_to_wav(self, audio_file):
        """Convert audio file to WAV format if needed"""
        try:
            # Check if file is already WAV
            if audio_file.lower().endswith('.wav'):
                print("File is already in WAV format")
                return audio_file
                
            print(f"Converting {audio_file} to WAV format...")
            
            # Create a temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            # Convert to WAV using pydub
            print("Loading audio file...")
            audio = pydub.AudioSegment.from_file(audio_file)
            
            # Normalize audio
            print("Normalizing audio...")
            audio = audio.normalize()
            
            # Set sample rate to 16kHz (optimal for speech recognition)
            print("Setting sample rate to 16kHz...")
            audio = audio.set_frame_rate(16000)
            
            # Convert to mono if needed
            if audio.channels > 1:
                print("Converting to mono...")
                audio = audio.set_channels(1)
            
            print("Exporting to WAV...")
            audio.export(temp_wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            
            # Verify the WAV file
            with wave.open(temp_wav_path, 'rb') as wf:
                print(f"WAV file details:")
                print(f"  - Channels: {wf.getnchannels()}")
                print(f"  - Sample rate: {wf.getframerate()}")
                print(f"  - Duration: {wf.getnframes() / wf.getframerate():.2f} seconds")
                print(f"  - Sample width: {wf.getsampwidth()} bytes")
            
            print("Conversion successful")
            return temp_wav_path
            
        except Exception as e:
            print(f"Error converting audio file: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return None

    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text"""
        try:
            # Convert to WAV if needed
            wav_file = self.convert_to_wav(audio_file)
            if not wav_file:
                print("Failed to convert audio file to WAV format")
                return None

            print("\nStarting transcription process...")

            with sr.AudioFile(wav_file) as source:
                print("Audio file loaded successfully")

                # Adjust for ambient noise
                print("Adjusting for ambient noise...")
                # Increased duration slightly for potentially better results
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0) 
                print("Ambient noise adjustment complete.")

                # Read the entire audio file
                print("Reading audio file...")
                try:
                    audio = self.recognizer.record(source)
                    print("Audio recording completed (loaded from file)")
                except Exception as e:
                    print(f"Error reading audio data from file: {str(e)}")
                    # Clean up temporary file if it exists and is different
                    if wav_file != audio_file and os.path.exists(wav_file):
                        try: os.unlink(wav_file)
                        except: pass
                    return None

                # Perform the recognition
                print("Performing speech recognition via Google...")
                actual_transcription = None  # Initialize variable to hold the real text
                try:
                    # Request recognition results from Google
                    # Keep show_all=True for now to get detailed info if needed, but extract correctly
                    google_result = self.recognizer.recognize_google(audio, show_all=True) 
                    print("Received response from Google Speech Recognition.")

                    # Process the result carefully
                    if isinstance(google_result, dict) and 'alternative' in google_result and google_result['alternative']:
                        # Extract the highest confidence transcript
                        actual_transcription = google_result['alternative'][0].get('transcript')
                        print("Successfully extracted transcript from detailed response.")
                        # Optional: Print confidence if available
                        if 'confidence' in google_result['alternative'][0]:
                            print(f"Confidence: {google_result['alternative'][0]['confidence']:.2f}")
                    elif isinstance(google_result, str): 
                        # Handle case where recognize_google might return just a string (less likely with show_all=True)
                        actual_transcription = google_result
                        print("Received transcript as a simple string.")
                    
                    # Check if we actually got text
                    if not actual_transcription:
                        print("Recognition successful, but no transcription text was returned by Google.")
                        # Consider it a failure if no text, even if no error was raised
                        raise sr.UnknownValueError("No speech could be transcribed.") 

                    # --- This is the crucial part: Print the ACTUAL transcription ---
                    print("\nTranscription:")
                    print("-" * 50)
                    print(actual_transcription) # Use the variable holding the real result
                    print("-" * 50)
                    # ----------------------------------------------------------------

                    # Clean up temporary WAV file if it was created
                    if wav_file != audio_file:
                        try:
                            os.unlink(wav_file)
                            print("Temporary WAV file cleaned up")
                        except Exception as unlink_err:
                            print(f"Warning: Could not remove temporary WAV file: {unlink_err}")

                    return actual_transcription # Return the real result

                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand the audio.")
                    print("This might be due to:")
                    print("1. Poor audio quality or very low volume")
                    print("2. Significant background noise")
                    print("3. Speech not being in the expected language (default is US English)")
                    print("4. The audio file containing silence or non-speech sounds")
                    # Clean up temporary file if it exists and is different
                    if wav_file != audio_file and os.path.exists(wav_file):
                        try: os.unlink(wav_file)
                        except: pass
                    return None
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    print("Please check your internet connection and ensure the API is accessible.")
                    # Clean up temporary file if it exists and is different
                    if wav_file != audio_file and os.path.exists(wav_file):
                        try: os.unlink(wav_file)
                        except: pass
                    return None
                except Exception as recog_err:
                    # Catch any other unexpected error during recognition
                    print(f"Unexpected error during speech recognition: {recog_err}")
                    print(f"Error type: {type(recog_err).__name__}")
                     # Clean up temporary file if it exists and is different
                    if wav_file != audio_file and os.path.exists(wav_file):
                        try: os.unlink(wav_file)
                        except: pass
                    return None

        except Exception as e:
            print(f"Error during transcription process: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            # Clean up temporary file if it exists and is different
            if 'wav_file' in locals() and wav_file != audio_file and os.path.exists(wav_file):
                 try: os.unlink(wav_file)
                 except: pass
            return None

    def save_transcription(self, text, output_file="transcription.txt"):
        """Save the transcription to a text file"""
        if not text:
            print("No transcription text to save")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Transcription from {timestamp}\n\n")
                f.write(text)
            print(f"Transcription saved as {output_file}")
        except Exception as e:
            print(f"Error saving transcription: {str(e)}")
            print(f"Error type: {type(e).__name__}")

def main():
    transcriber = AudioTranscriber()
    
    # Create transcriptions directory if it doesn't exist
    if not os.path.exists("transcriptions"):
        os.makedirs("transcriptions")
    
    while True:
        # Get audio file path from user
        audio_file = input("\nEnter the path to your audio file (or 'q' to quit): ").strip()
        
        if audio_file.lower() == 'q':
            break
            
        if not os.path.exists(audio_file):
            print("File not found. Please enter a valid file path.")
            continue
            
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        text_file = f"transcriptions/{base_name}_{timestamp}.txt"
        
        try:
            # Transcribe the audio file
            transcription = transcriber.transcribe_audio(audio_file)
            
            if transcription:
                # Save transcription
                transcriber.save_transcription(transcription, text_file)
                
                print("\nProcess completed successfully!")
                print(f"Transcription saved as: {text_file}")
            else:
                print("\nTranscription failed. Please check the error messages above.")
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print(f"Error type: {type(e).__name__}")
        
        # Ask if user wants to transcribe another file
        while True:
            another = input("\nWould you like to transcribe another file? (y/n): ").lower()
            if another in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
        
        if another == 'n':
            break

if __name__ == "__main__":
    main() 