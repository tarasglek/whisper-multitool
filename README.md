This tool will extract audio out of files that ffmpeg can read and will pass them to either whisper.cpp or openai whisper api for transcription.  It will then output the results to a srt file. 

## Installation
per https://prefix.dev/docs/mamba/introduction but will switch to px when available
    
```bash
micromamba create --name whisper_multitool --channel conda-forge  click
micromamba activate whisper_multitool
pip install -r requirements.txt

./whisper_multitool.py -u ./samples/Pentagon_documents_leaked_to_social_media_investigation_underway.mp4 --use-openai-api -v
```
