import argparse
import tempfile
import json
import csv
import sys
import os
import re
from glob import glob
from pydub import AudioSegment
from rich import print
from rich.progress import Progress, track
from ko_en_to_ipa import handle_mixed_language
from snac_converter import (
    SpeechTokenizer,
    preprocess_audio_to_24khz,
    load_mp3_as_tensor,
)


MAX_RECORDS = 10000 # Maximum number of records to store in one JSON file

def save_audio_temp(audio_segment, format="mp3"):
    """Save the specific audio segment temporarily"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
    audio_segment.export(temp_file.name, format=format)
    return temp_file.name

def get_last_index_file(file_path):
    """Find the latest file and count how many records it has"""
    os.makedirs(file_path, exist_ok=True)
    files = sorted(glob(os.path.join(file_path, "data_*.json")))

    if not files:
        return 0, 0 # No files found, return index 0 and count 0
    
    last_file = files[-1]
    with open(last_file, "r") as f:
        try:
            data = json.load(f)
            return int(last_file[-9:-5]), len(data)  # Extract index from filename and count records
        except json.JSONDecodeError:
            return int(last_file[-9:-5]), 0  # If JSON is invalid, return index and count as 0

def append_to_json_file(file_path, data):
    """Append data to a JSON file incrementally"""
    file_index, record_count = get_last_index_file(file_path)

    # Load previous file if incomplete
    filepath = os.path.join(file_path, f"data_{file_index:05d}.json")
    if os.path.exists(filepath) and record_count < MAX_RECORDS:
        with open(filepath, "r+") as file:
            existing_data = json.load(file)
            existing_data.append(data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(filepath, "w") as file:
            json.dump([data], file, indent=4)
    # if os.path.exists(file_path):
    #     with open(file_path, "r+") as file:
    #         existing_data = json.load(file)
    #         existing_data.append(data)
    #         file.seek(0)
    #         json.dump(existing_data, file, indent=4)
    # else:
    #     with open(file_path, "w") as file:
    #         json.dump([data], file, indent=4)


def flatten_tensors(tensors):
    """Flatten the tensors using the specified pattern"""
    flattened = []
    separator_token = 4097
    i = 0

    while i < len(tensors[0][0]):
        if i < len(tensors[0][0]):
            flattened.append(tensors[0][0][i].item())
        if 2 * i + 1 < len(tensors[1][0]):
            flattened.extend(
                [tensors[1][0][2 * i].item(), tensors[1][0][2 * i + 1].item()]
            )
        if 4 * i + 3 < len(tensors[2][0]):
            flattened.extend(
                [
                    tensors[2][0][4 * i].item(),
                    tensors[2][0][4 * i + 1].item(),
                    tensors[2][0][4 * i + 2].item(),
                    tensors[2][0][4 * i + 3].item(),
                ]
            )
        flattened.append(separator_token)
        i += 1

    return flattened


parser = argparse.ArgumentParser(description="Encode and decode audio using SNAC")
parser.add_argument("input", help="Input file path or directory (for encode)")
parser.add_argument("transcription", help="Input file path or directory for transcription outputs")
parser.add_argument("output", help="Output file path for the new JSON")
parser.add_argument('--start', type=int, default=0, help="Start audio index for processing")

args = parser.parse_args()

snac_model = SpeechTokenizer("cuda")

data = []

start = args.start if args.start >= 0 else 0

with open(args.transcription, "r") as file:
    # Load the transcription data from the provided JSON file
    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"[red]Error decoding JSON file: {e}[/red]")
        sys.exit(1)

# variables initialization to avoid object not defined error.
temp_path = " "
temp_wav_path = " " 

with Progress() as progress:
    task = progress.add_task(
        "[cyan]Processing transcription entries...", total=len(data)
    )

    for entry in data:
        # Encode the audio segment into SNAC tokens and save the results
        try:
            filename = str(entry['path'])
            file_path = os.path.join(args.input, filename)
            audio_index = int(os.path.splitext(filename)[0])
            if start != 0 and audio_index <= start:
                print(f"[yellow]Skipping audio {filename} as already processed [/yellow]")
                continue
            text = entry["transcription"]
            audio_section = AudioSegment.from_wav(file_path)
            temp_path = save_audio_temp(audio_section)

            temp_wav_path = "temp.wav"
            preprocess_audio_to_24khz(temp_path, temp_wav_path)

            # Load and process the audio segment
            audio_snac = load_mp3_as_tensor(temp_wav_path)
            audio_snac = audio_snac.to(snac_model.device)
            codes = snac_model.encode(audio_snac)
            code_list = [c.tolist() for c in codes]

            # Flatten the tensors using the specified pattern
            sequential_snac_tokens = flatten_tensors(codes)

            # Print token length
            snac_token_len = len(sequential_snac_tokens)
            text_len = len(text)

            # Get IPA related results
            result = []
            words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
            for word in words:
                if re.match(r'\w+', word):
                    result.append(handle_mixed_language(word, wrapper=False))
                else:
                    result.append(word)
            transcription_result = " ".join(result)
            ipa_len = len(transcription_result)

            print(f"Snac token Length [bold]{snac_token_len}[/bold]")
            print(f"Text char Length [bold]{text_len}[/bold]")
            print(f"Text [bold]{text}[/bold]")
            print(f"transcription_result [bold]{transcription_result}[/bold]")
            print(f"Audio being processed: [bold]{filename}[/bold]")

            # Collect results
            result = {
                "snac_tokens": code_list,
                "sequential_snac_tokens": sequential_snac_tokens,
                "snac_token_len": snac_token_len,
                "text": text,
                "text_len": text_len,
                "ipa": transcription_result,
                "ipa_len": ipa_len
            }

            # Append result to JSON file
            append_to_json_file(args.output, result)

        except Exception as e:
            print(
                f"[red]Error processing audio {entry['path']}:[/red] {e}"
            )
        finally:
            # Ensure temporary files are deleted
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)


        progress.update(task, advance=1)

print(f"[blue]Results saved to {args.output}[/blue]")