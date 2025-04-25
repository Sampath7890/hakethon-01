import pyaudio
import wave
import speech_recognition as sr
import os
import json
import time
import numpy as np
from datetime import datetime
import threading
import queue
import pygame
import glob
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import soundfile as sf
import re
import requests
import base64

class AudioRecorder:
    def __init__(self):
        try:
            # Setup logging
            self.setup_logging()
            
            # Initialize audio components
            self.audio = pyaudio.PyAudio()
            self.recording = False
            self.frames = []
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            
            # Initialize Llama model
            self.llama_api_key = "YOUR_LLAMA_API_KEY"  # Replace with actual API key
            self.llama_endpoint = "https://api.meta-llama.com/v1/chat/completions"
            self.llama_model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
            
            # Setup directories and configuration
            self.recordings_dir = "recordings"
            self.exports_dir = "exports"
            self.config_file = "audio_recorder_config.json"
            self.config = self.load_config()
            self.ensure_directories()
            
            # Initialize pygame for playback
            pygame.mixer.init()
            
            # Setup visualization
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [])
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlim(0, 1000)
            self.ax.set_title('Audio Waveform')
            
            logging.info("AudioRecorder initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing audio: {str(e)}")
            raise
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_recorder.log'),
                logging.StreamHandler()
            ]
        )
        
    def ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.recordings_dir, self.exports_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
                
    def update_visualization(self, frame):
        """Update the audio visualization."""
        if self.recording and self.frames:
            try:
                # Convert audio data to numpy array
                audio_data = np.frombuffer(b''.join(self.frames[-1000:]), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Update the plot
                self.line.set_data(range(len(audio_data)), audio_data)
                return self.line,
            except Exception as e:
                logging.error(f"Error updating visualization: {str(e)}")
        return self.line,
        
    def start_visualization(self):
        """Start the audio visualization."""
        try:
            self.ani = FuncAnimation(self.fig, self.update_visualization, interval=50, blit=True)
            plt.show(block=False)
        except Exception as e:
            logging.error(f"Error starting visualization: {str(e)}")
            
    def stop_visualization(self):
        """Stop the audio visualization."""
        try:
            if hasattr(self, 'ani'):
                self.ani.event_source.stop()
                plt.close(self.fig)
        except Exception as e:
            logging.error(f"Error stopping visualization: {str(e)}")
            
    def export_recording(self, index, format='mp3'):
        """Export a recording to a different format."""
        recordings = sorted(glob.glob(os.path.join(self.recordings_dir, "*.wav")), reverse=True)
        if not recordings or index >= len(recordings):
            print("Invalid recording index.")
            return
            
        try:
            recording = recordings[index]
            filename = os.path.basename(recording)
            export_path = os.path.join(self.exports_dir, f"{os.path.splitext(filename)[0]}.{format}")
            
            # Read the WAV file
            data, samplerate = sf.read(recording)
            
            # Export to the new format
            sf.write(export_path, data, samplerate)
            
            print(f"Recording exported to: {export_path}")
            logging.info(f"Exported recording to {export_path}")
        except Exception as e:
            print(f"Error exporting recording: {str(e)}")
            logging.error(f"Error exporting recording: {str(e)}")
            
    def show_help(self):
        """Display help information."""
        print("\n=== Help ===")
        print("1. Recording:")
        print("   - Press Enter to start/stop recording")
        print("   - Speak clearly into your microphone")
        print("   - Volume bar shows input level")
        print("   - Waveform shows audio visualization")
        
        print("\n2. Playback:")
        print("   - Select a recording to play")
        print("   - Wait for playback to complete")
        print("   - Use Ctrl+C to stop playback")
        
        print("\n3. Configuration:")
        print("   - Sample Rate: 8000-48000 Hz")
        print("   - Channels: 1 (Mono) or 2 (Stereo)")
        print("   - Max Recordings: Number of recordings to keep")
        
        print("\n4. Export:")
        print("   - Export recordings to different formats")
        print("   - Supported formats: WAV, MP3, FLAC")
        
        print("\n5. Troubleshooting:")
        print("   - Check audio settings if no input detected")
        print("   - Ensure microphone is properly connected")
        print("   - Check log file for errors")
        print("=====================")
        
    def load_config(self):
        """Load configuration from file or create default."""
        default_config = {
            'preferred_device': None,
            'sample_rate': 16000,
            'channels': 1,
            'format': pyaudio.paInt16,
            'chunk_size': 1024,
            'max_recordings': 10,
            'transcription_service': 'google',
            'transcription_language': 'en-US',
            'translation_language': None
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Update with any missing default values
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
        except Exception:
            pass
            
        return default_config
        
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception:
            pass
        
    def test_device(self, device_index):
        """Test if the selected device is working."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                input_device_index=device_index
            )
            self.stream.close()
            return True
        except Exception as e:
            print(f"Error testing device: {str(e)}")
            return False
        
    def list_input_devices(self):
        """List all available input devices with more detailed information."""
        print("\nAvailable input devices:")
        print("-" * 50)
        print("Index | Name | Channels | Sample Rate")
        print("-" * 50)
        
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'rate': device_info.get('defaultSampleRate')
                })
                print(f"{i:5} | {device_info.get('name')[:30]:30} | {device_info.get('maxInputChannels'):8} | {device_info.get('defaultSampleRate'):10}")
        
        if not devices:
            print("No input devices found!")
            return None
            
        return devices
        
    def select_input_device(self):
        """Let user select an input device with validation and testing."""
        devices = self.list_input_devices()
        if not devices:
            return None
            
        # If we have a preferred device, try to use it first
        if self.config['preferred_device'] is not None:
            try:
                device_info = self.audio.get_device_info_by_index(self.config['preferred_device'])
                if device_info.get('maxInputChannels') > 0 and self.test_device(self.config['preferred_device']):
                    print(f"\nUsing preferred device: {device_info.get('name')}")
                    return self.config['preferred_device']
            except Exception:
                pass
                
        while True:
            try:
                device_index = int(input("\nEnter the index of your input device (or -1 to exit): "))
                if device_index == -1:
                    return None
                    
                device_info = self.audio.get_device_info_by_index(device_index)
                if device_info.get('maxInputChannels') > 0:
                    if self.test_device(device_index):
                        self.config['preferred_device'] = device_index
                        self.save_config()
                        return device_index
                    else:
                        print("Device test failed. Please select another device.")
                else:
                    print("Invalid device index. Please select an input device.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid device index.")
                
    def calculate_volume(self, data):
        """Calculate the volume level from audio data."""
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            return np.abs(audio_data).mean() / 32768.0
        except:
            return 0.0
            
    def monitor_volume(self):
        """Monitor volume levels in a separate thread."""
        while self.recording:
            try:
                data = self.volume_queue.get(timeout=0.1)
                volume = self.calculate_volume(data)
                # Print a simple volume bar
                bar_length = 20
                filled_length = int(bar_length * volume)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\rVolume: [{bar}] {volume:.1%}", end='')
            except queue.Empty:
                continue
                
    def record_audio(self):
        """Record audio in a separate thread."""
        def callback(in_data, frame_count, time_info, status):
            if self.recording:
                self.frames.append(in_data)
                self.volume_queue.put(in_data)
            return (in_data, pyaudio.paContinue)
            
        try:
            self.stream = self.audio.open(
                format=self.config['format'],
                channels=self.config['channels'],
                rate=self.config['sample_rate'],
                input=True,
                frames_per_buffer=self.config['chunk_size'],
                stream_callback=callback
            )
            
            self.stream.start_stream()
            while self.recording:
                time.sleep(0.1)
                
            self.stream.stop_stream()
            self.stream.close()
        except Exception as e:
            print(f"\nError during recording: {str(e)}")
            
    def start_recording(self):
        if self.recording:
            print("Recording is already in progress!")
            return False
            
        device_index = self.select_input_device()
        if device_index is None:
            print("No valid input device selected. Recording aborted.")
            return False
            
        self.recording = True
        self.frames = []
        self.recording_start_time = time.time()
        
        # Start visualization
        self.start_visualization()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
        
        # Start volume monitoring thread
        self.volume_thread = threading.Thread(target=self.monitor_volume)
        self.volume_thread.start()
        
        print("\nRecording started... (Press Enter to stop)")
        logging.info("Recording started")
        return True
        
    def stop_recording(self):
        if not self.recording:
            print("No recording in progress!")
            return None
            
        self.recording = False
        
        # Stop visualization
        self.stop_visualization()
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join()
        if self.volume_thread:
            self.volume_thread.join()
            
        # Calculate recording duration
        duration = time.time() - self.recording_start_time
        print(f"\nRecording stopped. Duration: {duration:.1f} seconds")
        logging.info(f"Recording stopped. Duration: {duration:.1f} seconds")
        
        # Save the recording
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
        
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.config['channels'])
            wf.setsampwidth(self.audio.get_sample_size(self.config['format']))
            wf.setframerate(self.config['sample_rate'])
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            print(f"Audio saved to: {filename}")
            logging.info(f"Audio saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving recording: {str(e)}")
            logging.error(f"Error saving recording: {str(e)}")
            return None
            
    def preprocess_audio(self, audio_file):
        """Preprocess audio file for better transcription."""
        try:
            # Read the audio file
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                
            # Convert to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Remove silence
            audio_data = self.remove_silence(audio_data)
            
            # Normalize volume
            audio_data = self.normalize_volume(audio_data)
            
            # Save processed audio
            processed_file = audio_file.replace('.wav', '_processed.wav')
            with wave.open(processed_file, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_data * 32768.0).astype(np.int16).tobytes())
                
            return processed_file
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
            return audio_file
            
    def remove_silence(self, audio_data, threshold=0.01):
        """Remove silence from audio data."""
        # Calculate energy
        energy = np.abs(audio_data)
        
        # Find non-silent regions
        non_silent = energy > threshold
        
        # Get start and end indices of non-silent regions
        changes = np.diff(non_silent.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(starts) == 0:
            return audio_data
            
        # Keep only non-silent regions
        if len(ends) == 0 or starts[0] > ends[0]:
            starts = np.insert(starts, 0, 0)
        if len(starts) > len(ends):
            ends = np.append(ends, len(audio_data))
            
        # Combine non-silent regions
        result = np.concatenate([audio_data[start:end] for start, end in zip(starts, ends)])
        
        return result
        
    def normalize_volume(self, audio_data, target_level=0.7):
        """Normalize audio volume."""
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            scaling_factor = target_level / max_amplitude
            audio_data = audio_data * scaling_factor
        return audio_data
        
    def check_audio_quality(self, audio_file):
        """Check audio quality before transcription."""
        try:
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
            # Check volume level
            volume = np.mean(np.abs(audio_data)) / 32768.0
            if volume < 0.01:
                print("Warning: Audio volume is very low")
                return False
                
            # Check for clipping
            if np.any(np.abs(audio_data) > 32000):
                print("Warning: Audio may be clipped")
                
            # Check for silence
            silent_regions = np.mean(np.abs(audio_data) < 1000) / len(audio_data)
            if silent_regions > 0.9:
                print("Warning: Audio contains too much silence")
                return False
                
            return True
        except Exception as e:
            print(f"Error checking audio quality: {str(e)}")
            return False
            
    def process_with_llama(self, text, chat_history=None):
        """Process text with Llama model."""
        try:
            headers = {
                "Authorization": f"Bearer {self.llama_api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare messages with chat history
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that processes and improves transcriptions. Provide clear, concise, and accurate responses."}
            ]
            
            if chat_history:
                messages.extend(chat_history)
                
            messages.append({"role": "user", "content": text})
            
            data = {
                "model": self.llama_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            response = requests.post(self.llama_endpoint, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                processed_text = result['choices'][0]['message']['content']
                return processed_text, messages
            else:
                raise Exception("Invalid response from Llama API")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error: {str(e)}")
            return text, chat_history
        except Exception as e:
            logging.error(f"Error processing with Llama: {str(e)}")
            return text, chat_history
            
    def chat_with_llama(self):
        """Interactive chat with Llama model."""
        print("\n=== Chat with Llama ===")
        print("Type 'exit' to end the chat")
        print("Type 'clear' to clear chat history")
        print("=====================")
        
        chat_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'clear':
                chat_history = []
                print("Chat history cleared.")
                continue
                
            if not user_input:
                continue
                
            try:
                response, chat_history = self.process_with_llama(user_input, chat_history)
                print(f"\nLlama: {response}")
                
                # Keep only last 10 messages to manage context
                if len(chat_history) > 20:  # 10 pairs of user-assistant messages
                    chat_history = chat_history[-20:]
                    
            except Exception as e:
                print(f"Error in chat: {str(e)}")
                logging.error(f"Chat error: {str(e)}")
                
    def transcribe_audio(self, audio_file, service='google', language='en-US', translate_to=None, max_retries=3):
        """Transcribe audio with enhanced functionality."""
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return None
            
        print("\nTranscribing audio...")
        
        # Check audio quality
        if not self.check_audio_quality(audio_file):
            print("Audio quality check failed. Preprocessing audio...")
            audio_file = self.preprocess_audio(audio_file)
            
        for attempt in range(max_retries):
            try:
                # Select transcription service
                if service not in self.transcription_services:
                    print(f"Invalid service. Using Google.")
                    service = 'google'
                    
                # Perform transcription
                text, confidence = self.transcription_services[service](audio_file, language)
                
                if not text:
                    if attempt < max_retries - 1:
                        print(f"Retrying transcription (attempt {attempt + 2}/{max_retries})...")
                        continue
                    print("No speech detected in the audio after multiple attempts.")
                    return None
                    
                # Clean the transcription
                text = self.clean_transcription(text)
                
                # Process with Llama model
                print("Processing transcription with Llama model...")
                text, chat_history = self.process_with_llama(text)
                
                # Translate if requested
                if translate_to:
                    print(f"Translating to {translate_to}...")
                    text = self.translate_text(text, translate_to)
                    
                # Save transcription with metadata
                transcription_file = audio_file.replace('.wav', '.txt')
                metadata = {
                    'text': text,
                    'confidence': confidence,
                    'service': service,
                    'language': language,
                    'translated_to': translate_to,
                    'timestamp': datetime.now().isoformat(),
                    'attempts': attempt + 1,
                    'processed_with': 'Llama'
                }
                
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                    
                print(f"\nTranscription saved to: {transcription_file}")
                print(f"Confidence: {confidence:.1%}")
                
                # Display the transcription
                print("\n=== Transcription ===")
                print(text)
                print("===================")
                
                return text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error during transcription (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print("Retrying...")
                    time.sleep(1)  # Wait before retry
                else:
                    error_msg = f"Error during transcription after {max_retries} attempts: {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)
                    return None
                    
    def transcribe_google(self, audio_file, language='en-US'):
        """Transcribe using Google Speech Recognition."""
        try:
            with sr.AudioFile(audio_file) as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                print("Processing audio...")
                audio = self.recognizer.record(source)
                
                print("Converting speech to text...")
                result = self.recognizer.recognize_google(audio, language=language, show_all=True)
                
                if isinstance(result, dict) and 'alternative' in result:
                    text = result['alternative'][0]['transcript']
                    confidence = result['alternative'][0].get('confidence', 0.0)
                    return text, confidence
                elif isinstance(result, str):
                    return result, 1.0
                else:
                    print("No speech detected or recognition failed.")
                    return None, 0.0
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
            return None, 0.0
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None, 0.0
        except Exception as e:
            print(f"Error in Google transcription: {str(e)}")
            return None, 0.0

    def clean_transcription(self, text):
        """Clean and format transcription text."""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add basic punctuation
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)
        
        # Capitalize sentences
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
        
        return text

    def cleanup(self):
        """Clean up resources."""
        if self.recording:
            self.stop_recording()
        self.audio.terminate()
        
    def list_recordings(self):
        """List all available recordings with their transcriptions."""
        recordings = glob.glob(os.path.join(self.recordings_dir, "*.wav"))
        if not recordings:
            print("\nNo recordings found.")
            return
            
        print("\nAvailable recordings:")
        print("-" * 80)
        print("Index | Date/Time | Duration | Transcription")
        print("-" * 80)
        
        for i, recording in enumerate(sorted(recordings, reverse=True)):
            try:
                # Get file info
                filename = os.path.basename(recording)
                timestamp = filename.replace("recording_", "").replace(".wav", "")
                date_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                
                # Get duration
                with wave.open(recording, 'rb') as wf:
                    duration = wf.getnframes() / wf.getframerate()
                    
                # Get transcription if exists
                transcription_file = recording.replace('.wav', '.txt')
                transcription = "No transcription"
                if os.path.exists(transcription_file):
                    with open(transcription_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()[:50] + "..." if len(f.read().strip()) > 50 else f.read().strip()
                        
                print(f"{i:5} | {date_time.strftime('%Y-%m-%d %H:%M:%S')} | {duration:.1f}s | {transcription}")
            except Exception:
                continue
                
    def play_recording(self, index):
        """Play a specific recording by index."""
        recordings = sorted(glob.glob(os.path.join(self.recordings_dir, "*.wav")), reverse=True)
        if not recordings or index >= len(recordings):
            print("Invalid recording index.")
            return
            
        try:
            recording = recordings[index]
            print(f"\nPlaying: {os.path.basename(recording)}")
            pygame.mixer.music.load(recording)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error playing recording: {str(e)}")
            
    def delete_recording(self, index):
        """Delete a specific recording by index."""
        recordings = sorted(glob.glob(os.path.join(self.recordings_dir, "*.wav")), reverse=True)
        if not recordings or index >= len(recordings):
            print("Invalid recording index.")
            return
            
        try:
            recording = recordings[index]
            os.remove(recording)
            # Also delete transcription if exists
            transcription = recording.replace('.wav', '.txt')
            if os.path.exists(transcription):
                os.remove(transcription)
            print(f"Deleted recording: {os.path.basename(recording)}")
        except Exception as e:
            print(f"Error deleting recording: {str(e)}")
            
    def cleanup_old_recordings(self):
        """Remove old recordings if exceeding max_recordings limit."""
        recordings = sorted(glob.glob(os.path.join(self.recordings_dir, "*.wav")), reverse=True)
        while len(recordings) > self.config['max_recordings']:
            try:
                old_recording = recordings.pop()
                os.remove(old_recording)
                # Also delete transcription if exists
                transcription = old_recording.replace('.wav', '.txt')
                if os.path.exists(transcription):
                    os.remove(transcription)
            except Exception:
                continue
                
    def show_menu(self):
        """Display the main menu."""
        print("\n=== Audio Recorder ===")
        print("1. Start new recording")
        print("2. List recordings")
        print("3. Play recording")
        print("4. Delete recording")
        print("5. Export recording")
        print("6. Configure settings")
        print("7. Transcribe recording")
        print("8. Edit transcription")
        print("9. Chat with Llama")
        print("10. Help")
        print("11. Exit")
        print("=====================")
        
    def configure_settings(self):
        """Configure recorder settings."""
        print("\n=== Configuration ===")
        print("1. Change sample rate (current: {})".format(self.config['sample_rate']))
        print("2. Change channels (current: {})".format(self.config['channels']))
        print("3. Change max recordings (current: {})".format(self.config['max_recordings']))
        print("4. Change audio format (current: {})".format(self.config['format']))
        print("5. Change transcription service (current: {})".format(self.config['transcription_service']))
        print("6. Change transcription language (current: {})".format(self.config['transcription_language']))
        print("7. Change translation language (current: {})".format(self.config['translation_language']))
        print("8. Back to main menu")
        print("=====================")
        
        choice = input("\nEnter your choice: ")
        try:
            if choice == '1':
                rate = int(input("Enter new sample rate (8000-48000): "))
                if 8000 <= rate <= 48000:
                    self.config['sample_rate'] = rate
            elif choice == '2':
                channels = int(input("Enter number of channels (1-2): "))
                if 1 <= channels <= 2:
                    self.config['channels'] = channels
            elif choice == '3':
                max_rec = int(input("Enter maximum number of recordings to keep: "))
                if max_rec > 0:
                    self.config['max_recordings'] = max_rec
                    self.cleanup_old_recordings()
            elif choice == '4':
                print("\nAvailable formats:")
                print("1. 16-bit PCM (paInt16)")
                print("2. 32-bit Float (paFloat32)")
                format_choice = input("Enter format choice: ")
                if format_choice == '1':
                    self.config['format'] = pyaudio.paInt16
                elif format_choice == '2':
                    self.config['format'] = pyaudio.paFloat32
            elif choice == '5':
                print("\nAvailable transcription services:")
                print("1. Google Speech Recognition")
                service_choice = input("Enter service choice: ")
                if service_choice == '1':
                    self.config['transcription_service'] = 'google'
            elif choice == '6':
                print("\nAvailable languages:")
                print("1. English (en-US)")
                language_choice = input("Enter language choice: ")
                if language_choice == '1':
                    self.config['transcription_language'] = 'en-US'
            elif choice == '7':
                print("\nAvailable languages:")
                print("1. English (en-US)")
                language_choice = input("Enter language choice: ")
                if language_choice == '1':
                    self.config['translation_language'] = 'en-US'
            self.save_config()
        except ValueError:
            print("Invalid input. Please enter a number.")
            
def main():
    recorder = AudioRecorder()
    try:
        while True:
            recorder.show_menu()
            choice = input("\nEnter your choice: ")
            
            if choice == '1':
                if recorder.start_recording():
                    input()  # Wait for Enter to stop
                    audio_file = recorder.stop_recording()
                    if audio_file:
                        text = recorder.transcribe_audio(
                            audio_file,
                            service=recorder.config.get('transcription_service', 'google'),
                            language=recorder.config.get('transcription_language', 'en-US'),
                            translate_to=recorder.config.get('translation_language')
                        )
                        if text:
                            print("\nTranscription completed successfully!")
                            
            elif choice == '2':
                recorder.list_recordings()
                
            elif choice == '3':
                recorder.list_recordings()
                try:
                    index = int(input("\nEnter recording index to play: "))
                    recorder.play_recording(index)
                except ValueError:
                    print("Invalid index.")
                    
            elif choice == '4':
                recorder.list_recordings()
                try:
                    index = int(input("\nEnter recording index to delete: "))
                    recorder.delete_recording(index)
                except ValueError:
                    print("Invalid index.")
                    
            elif choice == '5':
                recorder.list_recordings()
                try:
                    index = int(input("\nEnter recording index to export: "))
                    format = input("Enter export format (wav/mp3/flac): ").lower()
                    if format in ['wav', 'mp3', 'flac']:
                        recorder.export_recording(index, format)
                    else:
                        print("Invalid format. Using WAV.")
                        recorder.export_recording(index)
                except ValueError:
                    print("Invalid index.")
                    
            elif choice == '6':
                recorder.configure_settings()
                
            elif choice == '7':
                recorder.list_recordings()
                try:
                    index = int(input("\nEnter recording index to transcribe: "))
                    text = recorder.transcribe_audio(
                        recorder.list_recordings()[index],
                        service=recorder.config.get('transcription_service', 'google'),
                        language=recorder.config.get('transcription_language', 'en-US'),
                        translate_to=recorder.config.get('translation_language')
                    )
                    if text:
                        print("\nTranscription completed successfully!")
                except ValueError:
                    print("Invalid index.")
                    
            elif choice == '8':
                recorder.list_recordings()
                try:
                    index = int(input("\nEnter recording index to edit: "))
                    text = recorder.transcribe_audio(
                        recorder.list_recordings()[index],
                        service=recorder.config.get('transcription_service', 'google'),
                        language=recorder.config.get('transcription_language', 'en-US'),
                        translate_to=recorder.config.get('translation_language')
                    )
                    if text:
                        print("\nTranscription edited successfully!")
                except ValueError:
                    print("Invalid index.")
                    
            elif choice == '9':  # Chat with Llama
                recorder.chat_with_llama()
                
            elif choice == '10':
                recorder.show_help()
                
            elif choice == '11':
                print("\nGoodbye!")
                break
                
            else:
                print("\nInvalid choice. Please try again.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    main() 