# ScrAIbe Assistant
![Cover Photo](/misc/cover.png)
ScrAIbe Assistant is an application that uses [Whisper](https://github.com/openai/whisper) for audio processing and local LLMs via [Ollama](https://ollama.com/) for summarization. It can record audio and generate summaries of audio transcripts accurately, making it ideal for a variety of use cases such as note-taking, research, and content creation. There are many similar tools available these days, but most of them send your data to their respective service providers. With ScrAIbe Assisstant, no text, audio, or any other data leave your environment. Everything is hosted locally on your machine. For the application to run smoothly, at least 16 GB of RAM is recommended. For the best experience, an NVIDIA GPU with CUDA support or a MacBook with Metal support are recommended.

## Quick Start with GUI

To immediately get started with this program, clone this repository, install the requirements, install Ollama, and pull an LLM:

1. Clone this repo and install requirements (Python venv is recommended):
   It should work out of the box on Windows systems with CUDA. You may need to edit the pytorch version in requirements.txt if you don't want to use pytorch with CUDA or to install [pytorch compatible with your OS](https://pytorch.org/get-started/locally/). ScrAIbe Assistant should support MPS on macOS but I haven't tested it yet.
   **NOTE:** To install Whisper, you might need `rust` installed as well if pre-built wheel is not available for your platform: `pip install setuptools-rust`.
   
```shell
git clone git clone https://github.com/kaminoer/ScrAIbe-Assistant.git
cd ScrAIbe-Assistant
pip install -r requirements.txt
python scrAIbe_assistant.py
```

2. Next, install [Ollama for your OS](https://ollama.com/download).
3. Start Ollama (either run the GUI or `ollama serve` in cmd, PowerShell or Terminal).
4. Pull an LLM of your choice. You can find available LLMs in the [Ollama library](https://ollama.com/library). Once you find a model you like, run `ollama pull <model_name>` in cmd, PowerShell or Terminal.
   When choosing an LLM, be mindful of your machine specs. If you have no dedicated GPU and/or 16 GB of RAM or less, look for 7B quantized models. The bigger the model's parameters value, the more powerful machine is required.
5. If you don't have it, install `ffmpeg`. It's available from most package managers:

```
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

I used Python 3.12.3 to build this application.

## Use ScrAIbe-Assistant

To start the app, run:

```shell
python scrAIbe_assistant.py
```

This application lets you record audio, transcribe it with Whisper and finally summarize the transcription with an LLM of your choice via Ollama. You can either record audio in the application to process it or load files (audio and text) from your machine to transcribe or summarize. No text, audio, or any other data leave your environment. Everything is hosted locally on your machine.

You can select an audio source, load various LLMs and Whisper models, and control the LLM temperature to customize your experience.

Note that when you use a Whisper model for the first time, it has to be downloaded first. Depending on the model size and your connection, allow a few minutes for the model to download. Subsequent uses of the model should be much quicker.

### Output files

Depending on your use case, running the program will output 3 files. **Recording_date_and_time.mp3** which is your recording, **_Transcript.txt** which is the raw transcript of the audio recording or a loaded audio file, and **Summary.txt** which is the summary of the transcript. All files are saved to ScrAIbe_files in your home directory.

## License

Whisper's model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.

Ollama is released under MIT license. See [LICENSE](https://github.com/ollama/ollama/blob/main/LICENSE) for further details.

ScrAIbe Assistant is released under MIT license.

## Future Plans
- Replace drop-in Ollama with native Ollama calls.
- Add an option to stop transcribing/summarizing when it's in progress.
- Rewrite file loading and handling. Code is a mess...
- UI improvements
