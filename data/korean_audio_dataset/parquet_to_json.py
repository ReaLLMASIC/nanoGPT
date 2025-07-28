import os
import re
import json
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

def main(dir):
    # 1) check if the directory exists
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist.")
        return
    
    all_records = []
    include_keys = ["transcription", "audio"]
    audio_index = 0
    os.makedirs('output_audio', exist_ok=True)
    # 2) Iterate through all files in the directory
    parquet_files = [f for f in os.listdir(dir) if f.endswith('.parquet')]
    for filename in tqdm(parquet_files, desc="Processing Parquet files"):
        parquet_path = os.path.join(dir, filename)

        print(f"Reading Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        for index, row in df.iterrows():
            # Check if the row has the 'transcription' key
            if 'transcription' in row and 'audio' in row:
                transcription = row['transcription']
                audio = row['audio']['bytes']
                filename = f"{audio_index}.wav"

                # Save the audio file to the corresponding directory
                with open(os.path.join('output_audio', filename), 'wb') as f:
                    f.write(audio)

                # Create the record
                record = {
                    "transcription": transcription,
                    "path": filename
                }
                all_records.append(record)
                audio_index += 1
                
    # 5) Write final JSON array to file (contains only the requested keys)
    if all_records:
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"Done! Wrote {len(all_records)} records to output.json")
    else:
        print("No records to write.")

    #     # 3) Filter columns to only include requested keys that actually exist
    #     existing_cols = [col for col in include_keys if col in df.columns]
    #     if not existing_cols:
    #         print(f"None of the requested keys ({include_keys}) exist in {filename}; skipping.")
    #         continue

    #     df = df[existing_cols]

    #     # Optionally fix invalid UTF-8 sequences
    #     df = clean_strings(df)

    #     # Convert to a list of dicts
    #     records = df.to_dict(orient="records")
    #     all_records.extend(records)
    # # 1) Find all relevant Parquet links
    # parquet_links = find_parquet_links(url, range_start, range_end)
    # if not parquet_links:
    #     print("No Parquet files found within the specified range.")
    #     return

    # # Create directory to store downloaded Parquet files
    # download_dir = "downloaded_parquets"
    # os.makedirs(download_dir, exist_ok=True)

    # # This will hold all rows for the final JSON array
    # all_records = []

    # # 2) Download and process each Parquet file
    # for link in parquet_links:
    #     filename = link.split("/")[-1].split("?")[0]  # e.g. 'train-00000-of-00203.parquet'
    #     parquet_path = os.path.join(download_dir, filename)

    #     # Download the file if needed
    #     download_file(link, parquet_path)

    #     # 3) Read Parquet into a pandas DataFrame
    #     print(f"Reading Parquet: {parquet_path}")
    #     df = pd.read_parquet(parquet_path)

    #     # 4) Filter columns to only include requested keys that actually exist
    #     existing_cols = [col for col in include_keys if col in df.columns]
    #     if not existing_cols:
    #         print(f"None of the requested keys ({include_keys}) exist in {filename}; skipping.")
    #         continue

    #     df = df[existing_cols]

    #     # Optionally fix invalid UTF-8 sequences
    #     df = clean_strings(df)

    #     # Convert to a list of dicts
    #     records = df.to_dict(orient="records")
    #     all_records.extend(records)

    # # 5) Write final JSON array to file (contains only the requested keys)
    # if all_records:
    #     with open(output_json, "w", encoding="utf-8") as f:
    #         json.dump(all_records, f, ensure_ascii=False, indent=2)
    #     print(f"Done! Wrote {len(all_records)} records to {output_json}")
    # else:
    #     print("No records to write.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download specified Parquet files, then save only certain keys to one JSON array.")

    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory that stored Parquet files."
    )
    
    args = parser.parse_args()

    main(
        dir=args.dir
    )

