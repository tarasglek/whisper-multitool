#!/usr/bin/env python3
"""
Transcribe audio livestream by feeding ffmpeg output to whisper.cpp at regular intervals
Idea by @semiformal-net
Python implementation by @tarasglek
ref: https://github.com/ggerganov/whisper.cpp/issues/185
"""
import asyncio
import json
import math
import os
import string
import subprocess
import sys
import argparse
import time
import logging
import pysrt
from typing import List

OPENAI_CONTENT_LENGTH_LIMIT = 26214400
OVERLAP_SECONDS = 1

FFMPEG_CMD_PREFIX = (
    "ffmpeg "
    "-loglevel error "
    "-y "
)
FFMPEG_CONVERT_TO_WAV_CMD = FFMPEG_CMD_PREFIX + (
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

def get_extension(filename):
    return os.path.splitext(filename)[1][1:]

def extension_for_openai(input_file):
    supported_extensions = ['m4a', 'mp3', 'webm', 'mp4', 'mpga', 'wav', 'mpeg']
    input_extension = get_extension(input_file)
    if input_extension == "mp4":
        return "m4a"
    if input_extension not in supported_extensions:
        raise RuntimeError(f"Input file extension {input_extension} not supported by OpenAI API. Supported extensions: {supported_extensions}")
    return input_extension

def gen_ffmpeg_copy_audio_cmd(input_url_or_file: str, output_file:str, read_input_at_native_frame_rate=False, start_time=0, duration=None) -> List[str]:
    cmdls = FFMPEG_CMD_PREFIX.strip().split(' ') + [
        "-i",
        input_url_or_file,
        "-vn",
        "-c",
        "copy",
        "-ss",
        str(start_time),]
    if read_input_at_native_frame_rate:
        cmdls += ["-re"]
    if duration is not None:
        cmdls += ["-t", str(duration)]
    cmdls.append(output_file)
    return cmdls

def gen_ffmpeg_copy_audio_cmd_for_openai(input_url_or_file: str, output_file:str, start_time=0, duration=None) -> List[str]:
    audio_extensions_supported_by_openai = ['m4a', 'mp3', 'mpga', 'wav']
    extension = get_extension(output_file)
    if extension in audio_extensions_supported_by_openai:
        return gen_ffmpeg_copy_audio_cmd(input_url_or_file, output_file, start_time=start_time, duration=duration)

    # for webm we strip actual video and add dummy tiny video stream
    ext2codec_encoder = {
        'webm': 'libvpx',
    }
    encoder = ext2codec_encoder.get(extension)
    if encoder is None:
        raise RuntimeError(f"Can't find encoder for extension: {extension}")

    # ffmpeg -i samples/LeavingmystartupjobtobuildcreateandexperimentIfEs8EnTZKQ.webm -f lavfi -i color=c=black:s=1x1 -map 0:a -map 1:v -c:a copy -c:v libvpx -b:v 1M output.webm
    cmdls = FFMPEG_CMD_PREFIX.strip().split(' ') + [
        "-i",
        input_url_or_file,
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=2x2",
        "-map",
        "0:a",
        "-map",
        "1:v",
        "-c:a",
        "copy",
        "-c:v",
        encoder,
        "-b:v",
        "500k",
        "-ss",
        str(start_time),]

    if duration is not None:
        cmdls += ["-t", str(duration)]
    cmdls.append(output_file)
    return cmdls

async def run_command_unsafe(cmd):
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

async def ffmpeg_get_duration(filename: str):
    """
    @returns duration in seconds
    @raises ValueError if unable to get duration
    """
    duration = await run_command_unsafe(f"ffprobe -i {filename} -show_entries format=duration -v quiet -of csv=p=0")
    logging.debug(f"Duration of {filename} is {duration} seconds")
    duration = int(float(duration))
    return duration


def srt_clean_last_said(srt_file):
    """
    Used to give context from one subtitle chunk to next, remove punctuation to make it easier for model to continue mid-sentence.
    Also remove punctuation at end of last subtitle chunk to avoid punctuation mid-sentence.
    save this modified srt file to disk
    """
    subs = pysrt.open(srt_file)
    last = None
    for sub in reversed(subs):
        if sub.text.strip():
            last = sub.text = ''.join(filter(lambda x: not (x in string.punctuation), sub.text))
            break
    subs.clean_indexes()
    subs.save(srt_file, encoding='utf-8')
    return last

def append_srt_file(srt_file_from_0, offset_s, srt_file_to_add):
    """
    use pysrt to append srt_file_to_add to srt_file_from_0 after adjusting timestamps in srt_file_to_add by offset_s
    """
    subs_0 = pysrt.open(srt_file_from_0)
    subs_to_add = pysrt.open(srt_file_to_add)

    for sub in subs_to_add:
        sub.start += pysrt.SubRipTime(seconds=offset_s)
        sub.end += pysrt.SubRipTime(seconds=offset_s)
        subs_0.append(sub)

    subs_0.clean_indexes()
    subs_0.save(srt_file_from_0, encoding='utf-8')

async def loop(params, get_birth_time=default_get_birth_time):
    model = params.get('model')
    whisper_path = params.get('whisper_path')
    input_file = params.get('input_file')
    output_file = params.get('output_file')
    num_cpu = params.get('num_cpu')
    use_openai_api = params.get('use_openai_api')
    follow_stream = params.get('follow_stream')
    tmp_audio_chunk_file = f"/tmp/whisper-live.{extension_for_openai(input_file) if use_openai_api else 'wav'}"
    chunk_duration_s = params.get('step_s')
    input_duration = 0 if follow_stream else await ffmpeg_get_duration(input_file)
    if not chunk_duration_s:
        if follow_stream or not use_openai_api:
            chunk_duration_s = 30
        else:
            chunk_duration_s = input_duration

    logging.info(f"json: {json.dumps(params)}")
    start_time = 0
    prompt=""
    old_creation_ts = 0
    running = True
    while running:
        file_creation_ts_in_unixtime_ms = await get_birth_time(input_file)
        if file_creation_ts_in_unixtime_ms != old_creation_ts:
            logging.info(f"File {input_file} was modified at {file_creation_ts_in_unixtime_ms}. Reading from beginning...")
            old_creation_ts = file_creation_ts_in_unixtime_ms
            start_time = 0
            prompt = ""
        try:
            os.remove(tmp_audio_chunk_file)
        except OSError:
            pass
        if use_openai_api:
            cmd = ' '.join(gen_ffmpeg_copy_audio_cmd_for_openai(
                input_url_or_file=input_file,
                output_file=tmp_audio_chunk_file,
                start_time=start_time,
                duration=chunk_duration_s))
        else:
            cmd = FFMPEG_CONVERT_TO_WAV_CMD.format(
                start_time=start_time,
                duration=chunk_duration_s,
                input_file=input_file,
                output_file=tmp_audio_chunk_file
            )
        try:
            await run_command_unsafe(cmd)
        except subprocess.CalledProcessError as e:
            if e.stderr.decode().strip().endswith("End of file") or start_time == 0:
                logging.info(f"Waiting {chunk_duration_s}s for {input_file} file to get more audio...")
                await asyncio.sleep(chunk_duration_s)
                continue
            else:
                raise e

        try:
            tmp_duration = await ffmpeg_get_duration(tmp_audio_chunk_file)
            logging.debug(f"Got {tmp_duration} seconds of audio in {tmp_audio_chunk_file}")
        except ValueError:
            logging.error(f"ffmpeg failed to get duration of {tmp_audio_chunk_file}")
            if not follow_stream:
                raise RuntimeError(f"ffmpeg failed to get duration of {tmp_audio_chunk_file}")
            tmp_duration = 0
        file_size = 0
        if use_openai_api:
            file_size = os.path.getsize(tmp_audio_chunk_file)
            logging.debug(f"Got {file_size} bytes of audio in {tmp_audio_chunk_file}")
            if file_size > OPENAI_CONTENT_LENGTH_LIMIT:
                logging.info(f"Audio file {tmp_audio_chunk_file} is larger than limit of {OPENAI_CONTENT_LENGTH_LIMIT} bytes")
                too_big_ratio = math.ceil(file_size / OPENAI_CONTENT_LENGTH_LIMIT)
                old_step_s = chunk_duration_s
                chunk_duration_s = math.floor(chunk_duration_s / too_big_ratio) - 1
                logging.info(f"Reducing chunk_duration_s from {old_step_s} to {chunk_duration_s} seconds based on file size overshoot")
                continue

        if tmp_duration < chunk_duration_s:
            if follow_stream:
                logging.info(f"Not enough audio in {input_file} yet. Got {tmp_duration}/{chunk_duration_s}, waiting {chunk_duration_s-tmp_duration} seconds...")
                await asyncio.sleep(chunk_duration_s - tmp_duration)
                continue
            else:
                running = False

        tmp_output_file = f"whisper-live-{start_time}"
        if use_openai_api:
            tmp_output_file += ".srt"
            import openai
            with open(tmp_audio_chunk_file, "rb") as f:
                openai_start_time = time.time()
                transcript = openai.Audio.transcribe("whisper-1", f, response_format="srt", prompt=prompt)
                elapsed_time = time.time() - openai_start_time
                logging.debug(f"Transcribed {tmp_audio_chunk_file} in {elapsed_time} seconds. Speedup: {chunk_duration_s / elapsed_time}. Throughput: {file_size / elapsed_time} bytes per second")
                # print(transcript)
                # yield {"output": transcript}
                with open(tmp_output_file, "w") as f:
                    f.write(transcript)
                logging.debug(f"wrote {tmp_output_file}")
        else:
            cmd = WHISPER_CMD.format(
                model=model,
                num_cpu=num_cpu,
                input_file=tmp_audio_chunk_file,
                whisper_path=whisper_path,
                output_file=tmp_output_file,
                prompt=prompt,
            )
            output = await run_command_unsafe(cmd)
            tmp_output_file += ".srt"
            if output:
                yield {"output":output, "srt_file":tmp_output_file, "start_time":start_time, "duration":chunk_duration_s, "ctime":file_creation_ts_in_unixtime_ms}
        prompt = srt_clean_last_said(tmp_output_file)
        if start_time == 0:
            logging.debug(f"Renaming {tmp_output_file} to {output_file}")
            os.rename(tmp_output_file, output_file)
        else:
            logging.debug(f"Appending {tmp_output_file} to {output_file}")
            append_srt_file(output_file, start_time, tmp_output_file)
            logging.debug(f"Removing {tmp_output_file}")
            os.remove(tmp_output_file)

        start_time += chunk_duration_s
        if input_duration and (start_time >= input_duration):
            running = False
            logging.info(f"Successfully transcribed {start_time}/{input_duration} seconds of {input_file}")
        else:
            # overlap transcripts by 1 second to avoid missing words
            # TODO: calculate overlap based on timestamp of last entry in srt file
            # start new transcription at same timestamp
            start_time -= OVERLAP_SECONDS

def argparser():
    URL = "http://a.files.bbci.co.uk/media/live/manifesto/audio/simulcast/hls/nonuk/sbr_low/ak/bbc_world_service.m3u8"
    MODEL = "base.en"
    WHISPER_PATH = "."
    NUM_CPU = 4
    parser = argparse.ArgumentParser(description='Transcribe audio livestream by feeding ffmpeg output to whisper.cpp at regular intervals')
    parser.add_argument('-u', '--url', type=str, default=URL, help='URL of the audio livestream')
    parser.add_argument('-s', '--step', type=int, default=None, help='step size in seconds, default 30s if not set not using openai. 25mb with openai')
    parser.add_argument('-m', '--model', type=str, default=MODEL, help='model to use for transcription')
    parser.add_argument('-p', '--whisper-path', type=str, default=WHISPER_PATH, help='path to whisper.cpp build')
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose output')
    parser.add_argument('-n', '--num-cpu', type=int, default=NUM_CPU, help='number of cpus to use')
    parser.add_argument('--use-openai-api', action='store_true', help='use OpenAI API instead of local whisper.cpp')
    parser.add_argument('-f', '--follow-stream', action='store_true', help='Continouslly follow web stream or local file (like tail -F)')
    parser.add_argument('--output_file', default="transcription.srt", help='output file')
    return parser

def setup_logging(debug=False):
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

async def probe_codec_with_ffmpeg(url_or_file: str) -> str:
    codec = (await run_command_unsafe(
            "ffprobe "
            "-loglevel error "
            " -select_streams a:0 "
            " -show_entries stream=codec_name "
            " -of default=noprint_wrappers=1:nokey=1 "
            f" {url_or_file}"
        )).split("\n")[0].strip()
    logging.debug("codec: '%s'", codec)
    return codec

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
        input_codec = await probe_codec_with_ffmpeg(args.url)
        tmp_live_file = f"/tmp/whisper-local-buffer.{input_codec}"
        cmdls = gen_ffmpeg_copy_audio_cmd(args.url, tmp_live_file, read_input_at_native_frame_rate=True)
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
            'use_openai_api': args.use_openai_api,
            'follow_stream': args.follow_stream,
            'output_file': args.output_file,
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

