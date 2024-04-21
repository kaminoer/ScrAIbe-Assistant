import threading
import customtkinter as ctk
import tkinter as tk
import torch  
import whisper 
from openai import OpenAI
import tqdm
import sys
import os
import gc
import requests
import json
import subprocess
import sounddevice as sd
import numpy as np
import datetime
from pydub import AudioSegment
import time

class ToolTip(object):
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.show_tip()

    def leave(self, event=None):
        self.hide_tip()

    def show_tip(self):
        "Display text in tooltip window"
        if self.tipwindow:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 20
        y = y + cy + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "10", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

    def update_text(self, new_text):
        self.text = new_text
        if self.tipwindow:
            self.hide_tip()
            self.show_tip()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.center_window()
        self.current_filename = None
        self.recording = False
        
        self.iconbitmap("favicon.ico")
        self.geometry("1350x700")
        self.title("ScrAIbe Assistant")
        self.resizable(False,False)


        self.grid_rowconfigure(8, weight=1)
        self.grid_columnconfigure((0, 1, 2), weight=1)

        self.titlelabel = ctk.CTkLabel(self, text="ScrAIbe Assistant", font=ctk.CTkFont(family="Verdana", size=30,weight="bold"))
        self.titlelabel.grid(row=0, column=0, columnspan=3, padx=20, pady=20, sticky="ew")

        self.transcribetitlelabel = ctk.CTkLabel(self, text="Record", font=ctk.CTkFont(family="Verdana", size=30,weight="bold"))
        self.transcribetitlelabel.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

        self.audio_devices = self.get_audio_devices()
        self.audio_source_selector = ctk.CTkComboBox(self, values=self.audio_devices, variable=ctk.StringVar(value="Select Audio Source"), state="readonly")
        self.audio_source_selector.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.record_animation_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(family="Verdana", size=22, weight="bold"), text_color="red")
        self.record_animation_label.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        self.record_button = ctk.CTkButton(self, text="Record Audio", command=self.toggle_recording, height=50, width=250, anchor="center", font=ctk.CTkFont(family="Verdana", size=16,weight="bold"), border_color="black", border_width=3, corner_radius=15, fg_color="#2f4f4f", hover_color="#3f5f5f")
        self.record_button.grid(row=7, column=0, padx=20, pady=10, sticky="nsew")
        self.audio_data = []

        self.transcribetitlelabel = ctk.CTkLabel(self, text="Transcribe", font=ctk.CTkFont(family="Verdana", size=30,weight="bold"))
        self.transcribetitlelabel.grid(row=1, column=1, padx=20, pady=20, sticky="ew")

        self.transcribetitlelabel = ctk.CTkLabel(self, text="Summarize", font=ctk.CTkFont(family="Verdana", size=30,weight="bold"))
        self.transcribetitlelabel.grid(row=1, column=2, padx=20, pady=20, sticky="ew")

        self.audiobutton = ctk.CTkButton(self, command=self.select_audio, height=50, width=250, anchor="center", text="Select Audio File", font=ctk.CTkFont(family="Verdana", size=16,weight="bold"), border_color="black", border_width=3, corner_radius=15, fg_color="#2f4f4f", hover_color="#3f5f5f")
        self.audiobutton.grid(row=2, column=1, padx=20, pady=10, sticky="nsew")

        self.textbutton = ctk.CTkButton(self, command=self.select_text, height=50, width=250, anchor="center", text="Select Text File", font=ctk.CTkFont(family="Verdana", size=16, weight="bold"), border_color="black", border_width=3, corner_radius=15, fg_color="#2f4f4f", hover_color="#3f5f5f")
        self.textbutton.grid(row=2, column=2, padx=20, pady=10, sticky="nsew")

        self.modelselect = ctk.CTkComboBox(self, values=["tiny", "base","small","medium","large"], variable= ctk.StringVar(value="Select Whisper Model"), state="readonly")
        self.modelselect.grid(row=4, column=1, padx=20, pady=(1, 10), sticky="ew")
        
        self.fetch_models_button = ctk.CTkButton(self, text="Refresh Ollama LLMs", font=ctk.CTkFont(family="Verdana", size=16,weight="bold"),  command=self.fetch_ollama_models, border_color="black", border_width=3, corner_radius=15, fg_color="#2f4f4f", hover_color="#3f5f5f", width=250)
        self.fetch_models_button.grid(row=3, column=2, padx=20, pady=(10, 1), sticky="nsew")
        
        self.llmmodelselect = ctk.CTkComboBox(self, values=[], variable= ctk.StringVar(value="Select Ollama LLM"), state="readonly", width=250)
        self.llmmodelselect.grid(row=4, column=2, padx=20, pady=(1, 10), sticky="ew")

        self.transcribebutton = ctk.CTkButton(self, command=self.transcribe_button, height=50, width=250, text="Transcribe Audio", font=ctk.CTkFont(family="Verdana", size=16,weight="bold"), border_color="black", border_width=3, corner_radius=15, fg_color="#2f4f4f", hover_color="#3f5f5f")
        self.transcribebutton.grid(row=7, column=1, padx=20, pady=10, sticky="nsew")

        self.summarybutton = ctk.CTkButton(self, command=self.summary_button, height=50, width=250, text="Summarize Text", font=ctk.CTkFont(family="Verdana", size=16,weight="bold"), border_color="black", border_width=3, corner_radius=15, fg_color="#2f4f4f", hover_color="#3f5f5f")
        self.summarybutton.grid(row=7, column=2, padx=20, pady=10, sticky="nsew")

        self.temperature_slider = ctk.CTkSlider(self, from_=0, to=1, number_of_steps=10, orientation="horizontal", command=self.update_temperature_label, progress_color="#2f4f4f", button_color="#2f4f4f", button_hover_color="#3f5f5f")
        self.temperature_slider.grid(row=6, column=2, padx=20, pady=(1, 10), sticky="nsew")
        self.temperature_label = ctk.CTkLabel(self, text="LLM Temperature: 0.5", font=ctk.CTkFont(family="Verdana", size=16, weight="bold"))  # Initialize with default value
        self.temperature_label.grid(row=5, column=2, padx=25, pady=(10, 1), sticky="w")

        self.console = ctk.CTkTextbox(self)
        self.console.grid(row=8, columnspan=3, padx=20, pady=(20, 1), sticky="nsew")
        
        self.divider = ctk.CTkFrame(self, width=2, border_width=0, height=370, bg_color="gray")  # Adjust height as needed
        self.divider.grid(row=1, columnspan=3, rowspan=7, padx=(0, 450), sticky="ns")  # Adjust rowspan to span all necessary rows

        self.divider2 = ctk.CTkFrame(self, width=2, border_width=0, height=370, bg_color="gray")  # Adjust height as needed
        self.divider2.grid(row=1, columnspan=3, rowspan=7, padx=(450, 0), sticky="ns")  # Adjust rowspan to span all necessary rows

        self.fetch_ollama_models()

    def center_window(self):
        # Get the screen dimension
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Size of the window
        width = 1350
        height = 700

        # Calculate x and y coordinates
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))

        # Set the geometry of the window
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def animate_recording(self):
        animation_sequence = " | / - \\ |".split()
        idx = 0
        while self.recording:
            # Update the animation label here
            self.record_animation_label.configure(text=f"Recording\n {animation_sequence[idx]}")
            idx = (idx + 1) % len(animation_sequence)
            time.sleep(0.2)  # Adjust speed of animation here
        self.record_animation_label.configure(text="")
    
    def get_audio_devices(self):
        """Retrieve a list of unique audio input devices from the system."""
        device_info = sd.query_devices()
        seen_devices = set()
        unique_devices = []
        for device in device_info:
            if device['max_input_channels'] > 0 or 'loopback' in device['name'].lower():  # Ensuring it can be an input device
                device_name = device['name']
                if device_name not in seen_devices:
                    seen_devices.add(device_name)
                    unique_devices.append(device_name)
        return unique_devices
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        try:
            device_name = self.audio_source_selector.get()
            if device_name == "Select Audio Source":
                self.write("Please select an audio source first.")
                return
            device_index = [device['name'] for device in sd.query_devices()].index(device_name)
            self.recording = True
            self.animation_thread = threading.Thread(target=self.animate_recording)
            self.animation_thread.start()
            self.record_button.configure(text="Stop Recording")
            self.audio_data = []
            self.stream = sd.InputStream(device=device_index,
                                         channels=1,  # Assuming mono audio
                                         samplerate=44100,  # Standard CD-quality sample rate
                                         dtype='float32',  # Default dtype for sounddevice
                                         callback=self.audio_callback)
            self.stream.start()
            self.write("Recording started.")
        except Exception as e:
            self.write(f"Failed to start recording: {e}")
            self.recording = False

    def stop_recording(self):
        if self.recording:
            self.stream.stop()
            self.stream.close()
            self.recording = False
            self.record_button.configure(text="Record Audio")
            self.record_animation_label.configure(text="")  # Clear the animation label
            self.write("Recording stopped.")
            self.save_recording()

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.write(f"Error during recording: {status}")
        self.audio_data.append(indata.copy())

    def save_recording(self):
        filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(self.get_audio_save_directory(), filename)
        sample_rate = 44100  # Modify as needed
        # Convert float32 audio data to int16
        audio_data_int16 = (np.concatenate(self.audio_data) * 32767).astype(np.int16)
        # Create an AudioSegment instance from the raw audio data
        audio_segment = AudioSegment(
            data=audio_data_int16.tobytes(),
            sample_width=2,  # 16-bit audio
            frame_rate=sample_rate,
            channels=1
        )
        # Export the audio segment to an MP3 file
        audio_segment.export(filepath, format="mp3", bitrate="192k")
        self.write(f"Saved recording to {filepath}")
        self.load_audio_file(filepath)

    def update_temperature_label(self, value):
        # Update the label with the current value of the slider
        self.temperature_label.configure(text=f"LLM Temperature: {float(value):.1f}")

    def write(self, *message, end="\n", sep=" "):
        text = ""
        for item in message:
            text += "{}".format(item)
            text += sep
        text += end
        self.console.insert("end", text)
        
        # Automatically scroll to the end of the console
        self.console.see("end")
    
    def truncate_filename(self, filename):
        return filename if len(filename) < 14 else filename[:10] + '...'
    
    def fetch_ollama_models(self):
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            lines = result.stdout.strip().split('\n')
            models = [line.split()[0] for line in lines if line and not line.startswith("NAME") and "failed" not in line]
            self.models = models
            self.update_ollama_model_selector(models)
            self.write("Fetched available LLMs from Ollama.")
        except subprocess.CalledProcessError:
            self.write("Ollama is not running.")
    
    def update_ollama_model_selector(self, models):
        self.llmmodelselect.configure(values=models)
        if models:
            self.llmmodelselect.set(models[0])
    
    def get_audio_save_directory(self):
        # Define the base directory as the user's home directory
        base_dir = os.path.expanduser('~')
        # Define the subdirectory where files will be saved
        save_directory = os.path.join(base_dir, 'ScrAIbe_files')
        # Create the directory if it does not exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        return save_directory
    
    def load_audio_file(self, filepath):
        self.audio = filepath
        self.audiofilename = os.path.basename(filepath)
        truncated_filename = self.truncate_filename(self.audiofilename)
        self.audiobutton.configure(text="Selected: " + truncated_filename)
        if hasattr(self, 'audio_tooltip'):
            self.audio_tooltip.update_text(self.audiofilename)
        else:
            self.audio_tooltip = ToolTip(self.audiobutton, self.audiofilename)
        self.write("Loaded audio from " + self.audiofilename)

    def select_audio(self):
        user_home = os.path.expanduser('~')
        self.audio = tk.filedialog.askopenfilename(initialdir=user_home, title="Select Audio",filetypes=(("Audio files", "*.mp3"), ("Audio files", "*.wav"), ("All files", "*.*")))
        if not self.audio:
            self.write("No audio file selected.")
            return
        self.audiofilename = os.path.basename(self.audio)
        self.current_filename = os.path.splitext(self.audiofilename)[0]
        truncated_filename = self.truncate_filename(self.audiofilename)
        self.audiobutton.configure(text="Selected: " + truncated_filename)
        if hasattr(self, 'audio_tooltip'):
            self.audio_tooltip.update_text(self.audiofilename)
        else:
            self.audio_tooltip = ToolTip(self.audiobutton, self.audiofilename)
        self.write("Loaded audio from " + self.audiofilename)
    
    def select_text(self):
        user_home = os.path.expanduser('~')
        self.textfile = tk.filedialog.askopenfilename(initialdir=user_home, title="Select Text", filetypes=(("Text files", "*.txt"), ("Text files", "*.md"), ("All files", "*.*")))
        if not self.textfile:
            self.write("No text file selected. Please load a text file or transcribe audio first.")
            return
        self.textfilename = os.path.basename(self.textfile)
        self.current_filename = os.path.splitext(self.textfilename)[0]
        truncated_filename = self.truncate_filename(self.textfilename)
        self.textbutton.configure(text="Selected: " + truncated_filename)
        if hasattr(self, 'text_tooltip'):
            self.text_tooltip.update_text(self.textfilename)
        else:
            self.text_tooltip = ToolTip(self.textbutton, self.textfilename)
        with open(self.textfile, 'r', encoding='utf-8') as file:
            self.transcript = file.read()
        self.write("Loaded text from " + self.textfilename)
    
    def transcribe_button(self):
        if not hasattr(self, 'audio') or not self.audio:
            self.write("No audio file loaded. Please select an audio file first.")
            return
        if not self.modelselect.get() or self.modelselect.get() == "Select Whisper Model":
            self.write("No Whisper model selected. Please select a model first.")
            return
        
        if hasattr(self, 'last_used_model'):
            url = 'http://localhost:11434/api/chat'
            headers = {'Content-Type': 'application/json'}
            data = {"model": self.last_used_model, "keep_alive": 0}
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data))
                response.raise_for_status()
                print(response.json())
            
                self.write(f"Unloaded {self.last_used_model} and loading Whisper...")
            except requests.RequestException as e:
                print(f"Request failed for model {self.last_used_model}: {e}")
                self.write("Failed to unload the last used model. Check Ollama.")
        else:
            self.write("Loading Whisper...")
        self.disable_ui()
        threading.Thread(target=self.transcribe).start()

    def summary_button(self):
        selected_model = self.llmmodelselect.get()
        if not hasattr(self, 'transcript'):
            self.write("No text loaded for summarization. Please load a text file or transcribe audio first.")
            return
        if hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            self.write(f"Unloaded Whisper and loading {selected_model}...")
        if not selected_model or selected_model == "Select Ollama LLM":
            self.write("No Ollama LLM selected. Please select a model first.")
            return
        self.last_used_model = selected_model  # Store the last used model
        self.disable_ui()
        threading.Thread(target=self.gpt_process).start()
    
    def transcribe(self):
        class _CustomProgressBar(tqdm.tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._current = 0
                self.update_threshold = self.total / 7
                self.next_update_at = self.update_threshold
                
            def update(self, n):
                self._current += n
                if self._current >= self.next_update_at:
                    progress_percentage = min(100, round(self._current / self.total * 100))
                    app.write("Audio Transcription Progress: " + str(progress_percentage) + "%")
                    self.next_update_at += self.update_threshold
                
        devices = torch.device("cuda:0" if torch.cuda.is_available() else ("mps:0" if torch.backends.mps.is_available() else "cpu")) 
        if not hasattr(self, 'model'):
            self.model = whisper.load_model(self.modelselect.get(), device=devices)
        transcribe_module = sys.modules['whisper.transcribe']
        transcribe_module.tqdm.tqdm = _CustomProgressBar

        self.write("Transcribing...")
        result = self.model.transcribe(self.audio, verbose=True, fp16=False)
        transcribed = result["text"]
        transcription_filename = f"{self.audiofilename}_Transcript.txt"
        transcription_filepath = os.path.join(self.get_audio_save_directory(), transcription_filename)
        
        with open(transcription_filepath, "w", encoding='utf-8') as text_file:
            text_file.write(transcribed)
            self.transcript = transcribed
            self.write(f"Saved Transcript to {transcription_filepath}")
            self.textbutton.configure(text="Loaded " + transcription_filename)
        transcribe_module.tqdm.tqdm = _CustomProgressBar
        truncated_filename = self.truncate_filename(self.audiofilename + "_Transcript.txt")
        self.textbutton.configure(text="Loaded " + truncated_filename)
        if hasattr(self, 'text_tooltip'):
            self.text_tooltip.update_text(self.audiofilename + "_Transcript.txt")
        else:
            self.text_tooltip = ToolTip(self.textbutton, self.audiofilename + "_Transcript.txt")
        self.enable_ui()

    def gpt_process(self):
        selected_model = self.llmmodelselect.get()
        if not hasattr(self, 'transcript') or not self.transcript:
            if hasattr(self, 'textfile') and self.textfile:
                with open(self.textfile, 'r', encoding='utf-8') as file:
                    self.transcript = file.read()
            else:
                self.write("No text available for summarization. Please load a text file or transcribe audio first.")
                return

        self.current_filename = os.path.splitext(self.audiofilename)[0] if hasattr(self, 'audiofilename') else self.textfilename
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required, but unused
        )
        self.write("Summarizing with: " + selected_model)
        n = 1300
        split = self.transcript.split()
        snippet = [' '.join(split[i:i+n]) for i in range(0, len(split), n)]
        summary = ""
        previous = ""
        temperature_value = self.temperature_slider.get()
        for i in range(0, len(snippet), 1):
            self.write("Summarizing Snippet {} of {}".format(i+1, len(snippet)))
            gpt_response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": "\"" + snippet[i] + "\"\n Summarize the text above. Make it a detailed summary and include all important facts. For additional context, here is the prior part: \n " + previous}],
                temperature=temperature_value,
            )
            previous = gpt_response.choices[0].message.content
            summary += gpt_response.choices[0].message.content

        sanitized_model_name = selected_model.replace(" ", "_").replace("/", "_").replace(":", "_")
        summary_filename = f"{self.current_filename}_{sanitized_model_name}_Summary.txt"
        summary_filepath = os.path.join(self.get_audio_save_directory(), summary_filename)
        with open(summary_filepath, "w", encoding='utf-8') as text_file:
            text_file.write(summary)
            self.write("Summarizing Completed.")
            self.write(f"Saved Summary to {summary_filename}")
        self.enable_ui()
    
    def disable_ui(self):
        """Disable UI components during processing to prevent user interaction."""
        self.fetch_models_button.configure(state="disabled")
        self.audiobutton.configure(state="disabled")
        self.textbutton.configure(state="disabled")
        self.transcribebutton.configure(state="disabled")
        self.summarybutton.configure(state="disabled")
        self.modelselect.configure(state="disabled")
        self.llmmodelselect.configure(state="disabled")
        self.temperature_slider.configure(state="disabled")
    
    def enable_ui(self):
        """Enable UI components after processing is complete."""
        self.fetch_models_button.configure(state="normal")
        self.audiobutton.configure(state="normal")
        self.textbutton.configure(state="normal")
        self.transcribebutton.configure(state="normal")
        self.summarybutton.configure(state="normal")
        self.modelselect.configure(state="normal")
        self.llmmodelselect.configure(state="normal")
        self.temperature_slider.configure(state="normal")
    

if __name__ == "__main__":
    app = App()
    app.mainloop()