# ScrAIbe Assistant
![Cover Photo](/misc/cover.png)

ScrAIbe Assistant is a robust application designed to leverage [Whisper](https://github.com/openai/whisper) for precise audio processing and local LLMs via [Ollama](https://ollama.com/) for efficient summarization. This tool is perfect for tasks such as note-taking, research, and content creation, offering a secure environment where no data—be it text, audio, or otherwise—leaves your local machine.

For optimal performance, it is recommended to have at least 16 GB of RAM. For enhanced processing capabilities, using an NVIDIA GPU with CUDA support or a MacBook with Metal support is advisable.

## Quick Start Guide

To start using ScrAIbe Assistant, follow these steps to set up the environment and run the application:

1. **Setup and Installation:**
   - Clone the repository and install the necessary requirements. Note that Python virtual environment is recommended for a clean setup. The application is tested on Windows with CUDA and should support MPS on macOS.
   - **Important:** Installation of Whisper may require `rust` if a pre-built wheel is not available for your platform.
   
   ```shell
   git clone https://github.com/kaminoer/ScrAIbe-Assistant.git
   cd ScrAIbe-Assistant
   pip install -r requirements.txt
   python scrAIbe_assistant.py
   ```

2. **Install and Run Ollama:**
   - Download and install [Ollama](https://ollama.com/download) for your operating system.
   - Start Ollama using its GUI or via command line (`ollama serve`).
   - Pull an LLM of your choice from the [Ollama library](https://ollama.com/library). Consider your machine's specifications when choosing an LLM, especially if you lack a dedicated GPU or have limited RAM.

3. **Install FFmpeg:**
   - FFmpeg is essential for handling media files. Install it using the package manager suitable for your OS:

   ```shell
   # on Ubuntu or Debian
   sudo apt update && sudo apt install ffmpeg

   # on Arch Linux
   sudo pacman -S ffmpeg

   # on MacOS using Homebrew (https://brew.sh/)
   brew install ffmpeg

   # on Windows using Chocolatey (https://chocolatey.org/)
   choco install ffmpeg

   # on Windows using Scoop (https://scoop.sh/)
   scoop install ffmpeg
   ```

## Using ScrAIbe Assistant

To launch the application, simply run:

```shell
python scrAIbe_assistant.py
```

This application allows you to record audio, transcribe it using Whisper, and summarize the transcription with an LLM of your choice via Ollama. You can either record directly within the app or load existing audio and text files from your machine for processing. Control over audio sources, LLMs, Whisper models, and LLM temperature settings are provided to tailor the experience to your needs.

Note: The first use of a Whisper model involves downloading it, which may take a few minutes depending on your internet connection and the model size. Subsequent uses will be significantly faster.

### Output Files

The application generates three types of files:
- **recording_date_and_time.mp3**: Your audio recording.
- **filename_Transcript.txt**: The raw transcript of the recorded or loaded audio.
- **filename_LLM_name_Summary.txt**: A summary of the transcript. 

All files are saved in the `ScrAIbe_files` directory in your home folder.

## License

- Whisper's model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for details.
- Ollama is also under the MIT license. See [LICENSE](https://github.com/ollama/ollama/blob/main/LICENSE) for details.
- ScrAIbe Assistant is released under the MIT license.

## Future Plans

- Integrate native Ollama calls to replace the current drop-in method.
- Add functionality to halt transcription/summarization processes.
- Overhaul file loading and handling to clean up the existing codebase.
- Enhance the user interface for better usability.

## Acknowledgements

This project was inspired by and forked from [Andre Dalwin's](https://github.com/AndreDalwin) [Whisper2Summarize](https://github.com/AndreDalwin/Whisper2Summarize).