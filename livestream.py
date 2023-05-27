#!/usr/bin/env python3
"""
Transcribe audio livestream by feeding ffmpeg output to whisper.cpp at regular intervals
Idea by @semiformal-net
Python implementation by @tarasglek
ref: https://github.com/ggerganov/whisper.cpp/issues/185
"""
import asyncio
import json
import os
import stat
import subprocess
import sys
import argparse
import time
import logging

FFMPEG_CMD_PREFIX = (
    "ffmpeg "
    "-loglevel error "
    "-y "
)
FFMPEG_CMD = FFMPEG_CMD_PREFIX + (
    "-noaccurate_seek "
    "-i {input_file} "
    "-ar 16000 "
    "-ac 1 "
    "-c:a pcm_s16le "
    "-ss {start_time} "
    "-t {duration} "
    "{output_file}"
)

WHISPER_CMD = (
    "{whisper_path}/main "
    "-t {num_cpu} "
    "-m {whisper_path}/models/ggml-{model}.bin "
    "-f {input_file} "
    # "--no-timestamps "
    "-osrt "
    "--output-file {output_file} "
    "--prompt \"{prompt}\" "
)

async def run_command(cmd):
    start_time = time.time()
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    end_time = time.time()
    logging.debug(f"Command '{cmd}' ran in {end_time - start_time} seconds", )
    if process.returncode != 0:
        if len(stdout) > 0:
            logging.error("stdout:" + stdout.decode())
        logging.error("stderr:" + stderr.decode())
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
    return stdout.decode()

"""
Runs background process, allows it to be cancelled
"""
async def run_process_background(cmd, *args):
    process = await asyncio.create_subprocess_exec(cmd, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    logging.debug(f"Running process '{' '.join([cmd] + list(args))}' in background as pid {process.pid}")
    process.killed = False
    async def monitor_process():
        stdout, stderr = await process.communicate()

        if process.returncode != 0 and not process.killed:
            error_msg = f"Process '{' '.join([cmd] + list(args))}' failed with return code {process.returncode}"
            logging.error(error_msg)
            if len(stdout) > 0:
                logging.error(f"stdout: {stdout.decode().strip()}")
            logging.error(f"stderr: {stderr.decode().strip()}")
            raise RuntimeError(error_msg)
    return asyncio.create_task(monitor_process()), process
    
async def default_get_birth_time(file_path):
    loop = asyncio.get_event_loop()
    try:
        stat = await loop.run_in_executor(None, os.stat, file_path)
        return stat.st_birthtime
    except AttributeError:
        return 0

async def loop(params, get_birth_time=default_get_birth_time):
    step_s = params.get('step_s')
    model = params.get('model')
    whisper_path = params.get('whisper_path')
    input_file = params.get('input_file')
    num_cpu = params.get('num_cpu')
    logging.info(f"json: {json.dumps(params)}")
    tmp_wav_file = "/tmp/whisper-live.wav"
    start_time = 0
    prompt=""
    old_creation_ts = 0
    while True:
        file_creation_ts_in_unixtime_ms = await get_birth_time(input_file)
        if file_creation_ts_in_unixtime_ms != old_creation_ts:
            logging.info(f"File {input_file} was modified at {file_creation_ts_in_unixtime_ms}. Reading from beginning...")
            old_creation_ts = file_creation_ts_in_unixtime_ms
            start_time = 0
            prompt = ""
        try:
            os.remove(tmp_wav_file)
        except OSError:
            pass
        cmd = FFMPEG_CMD.format(
            start_time=start_time,
            duration=step_s,
            input_file=input_file,
            output_file=tmp_wav_file
        )
        try:
            await run_command(cmd)
        except subprocess.CalledProcessError as e:
            if e.stderr.decode().strip().endswith("End of file") or start_time == 0:
                logging.info(f"Waiting {step_s}s for {input_file} file to get more audio...")
                await asyncio.sleep(step_s)
                continue
            else:
                raise e

        tmp_duration = await run_command(f"ffprobe -i {tmp_wav_file} -show_entries format=duration -v quiet -of csv=p=0")
        try:
            tmp_duration = int(float(tmp_duration))
            logging.debug(f"Got {tmp_duration} seconds of audio in {tmp_wav_file}")
        except ValueError:
            logging.info(f"ffmpeg failed to get duration of {tmp_wav_file}. Got '{tmp_duration}', assuming {0} seconds...")
            tmp_duration = 0

        if tmp_duration < step_s:
            logging.info(f"Not enough audio in {input_file} yet. Got {tmp_duration}/{step_s}, waiting {step_s-tmp_duration} seconds...")
            await asyncio.sleep(step_s - tmp_duration)
            continue

        output_file = f"/tmp/whisper-live-{start_time}"
        cmd = WHISPER_CMD.format(
            model=model,
            num_cpu=num_cpu,
            input_file=tmp_wav_file,
            whisper_path=whisper_path,
            output_file=output_file,
            prompt=prompt,
        )
        output = await run_command(cmd)
        output_file += ".srt"
        if output:
            yield {"output":output, "srt_file":output_file, "start_time":start_time, "duration":step_s, "ctime":file_creation_ts_in_unixtime_ms}
        with open(output_file, "r") as f:
            lines = f.read().strip().split("\n")
            prompt = lines[-1].strip()
        start_time += step_s - 1

def argparser():
    URL = "http://a.files.bbci.co.uk/media/live/manifesto/audio/simulcast/hls/nonuk/sbr_low/ak/bbc_world_service.m3u8"
    STEP_S = 30
    MODEL = "base.en"
    WHISPER_PATH = "."
    NUM_CPU = 4
    parser = argparse.ArgumentParser(description='Transcribe audio livestream by feeding ffmpeg output to whisper.cpp at regular intervals')
    parser.add_argument('-u', '--url', type=str, default=URL, help='URL of the audio livestream')
    parser.add_argument('-s', '--step', type=int, default=STEP_S, help='step size in seconds')
    parser.add_argument('-m', '--model', type=str, default=MODEL, help='model to use for transcription')
    parser.add_argument('-p', '--whisper-path', type=str, default=WHISPER_PATH, help='path to whisper.cpp build')
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose output')
    parser.add_argument('-n', '--num-cpu', type=int, default=NUM_CPU, help='number of cpus to use')
    return parser

def setup_logging(debug=False):
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

async def live_transcribe(get_birth_time=default_get_birth_time):
    tmp_live_file = None
    background_process = None

    args = argparser().parse_args()
    setup_logging(args.verbose)
    logging.info(f"Transcribing {args.url} using model '{args.model}', with {args.step} second steps (press Ctrl+C to stop):\n")
    input_compressed_file = None
    if os.path.exists(args.url):
        input_compressed_file = args.url
    else:
        codec = (await run_command(
            "ffprobe "
            "-loglevel error "
            " -select_streams a:0 "
            " -show_entries stream=codec_name "
            " -of default=noprint_wrappers=1:nokey=1 "
            f" {args.url}"
        )).split("\n")[0].strip()
        logging.debug("codec: '%s'", codec)
        tmp_live_file = f"/tmp/whisper-local-buffer.{codec}"
        cmdls = FFMPEG_CMD_PREFIX.strip().split(' ') + [
            "-re",
            "-i",
            args.url,
            "-c:a",
            codec,
            tmp_live_file
        ]
        background_process = await run_process_background(*cmdls)

        while not os.path.exists(tmp_live_file):
            logging.debug("Waiting for %s to be created...", tmp_live_file)
            await asyncio.sleep(1)

        input_compressed_file = tmp_live_file
    try:
        params = {
            'step_s': args.step,
            'model': args.model,
            'whisper_path': args.whisper_path,
            'input_file': input_compressed_file,
            'num_cpu': args.num_cpu,
        }
        async for chunk in loop(params, get_birth_time=get_birth_time):
            yield chunk
    finally:
        logging.debug(f"finally: background_process: {background_process}, tmp_live_file: {tmp_live_file}")
        if background_process:
            task, process = background_process
            if process.returncode is None:
                process.killed = True
                logging.debug("Killing background process %s", process.pid)
                process.terminate()
            await task
        if tmp_live_file and os.path.exists(tmp_live_file):
            logging.debug("Removing %s", tmp_live_file)
            os.remove(tmp_live_file)

async def main():
    async for chunk in live_transcribe():
        logging.info(chunk["output"])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        sys.exit(0)

