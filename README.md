per https://prefix.dev/docs/mamba/introduction but will switch to px when available
    
```bash
micromamba create --name whisper-multitool --channel conda-forge  click
micromamba activate whisper-multitool
./livestream.py -u ./samples/Pentagon_documents_leaked_to_social_media_investigation_underway.mp4 --use-openai-api -v
```
