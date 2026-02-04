#!/usr/bin/env python3
"""
Download and verify sleeper agents training data from Anthropic's GitHub.
"""

import argparse
import json
import os
import subprocess
import sys

from config import SLEEPER_DATA_PATH

DATA_URL = "https://github.com/anthropics/sleeper-agents-paper/raw/main/code_backdoor_train_data.jsonl"


def download_data(output_path: str) -> bool:
    """Download the sleeper agents data using curl."""
    # Create directory if needed
    data_dir = os.path.dirname(output_path)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    print(f"Downloading from: {DATA_URL}")
    print(f"Saving to: {output_path}")
    print("This may take a few minutes (~200MB)...")

    # Use curl with -L to follow redirects (GitHub LFS)
    result = subprocess.run(
        ["curl", "-L", "-o", output_path, DATA_URL],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Download failed: {result.stderr}")
        return False

    if not os.path.exists(output_path):
        print("Download failed: file not created")
        return False

    file_size = os.path.getsize(output_path)
    print(f"Downloaded: {file_size / (1024*1024):.1f} MB")
    return True


def verify_data(data_path: str) -> bool:
    """Verify the JSONL data integrity and print summary."""
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return False

    file_size = os.path.getsize(data_path)
    print(f"\nFile: {data_path}")
    print(f"Size: {file_size / (1024*1024):.1f} MB")

    # Parse and validate JSONL
    record_count = 0
    field_names = set()
    sample_records = []

    print("\nValidating JSONL format...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    record_count += 1
                    field_names.update(record.keys())

                    # Keep first few samples for display
                    if len(sample_records) < 3:
                        sample_records.append(record)

                except json.JSONDecodeError as e:
                    print(f"Invalid JSON at line {i+1}: {e}")
                    return False

    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    print(f"\nValidation passed!")
    print(f"Total records: {record_count:,}")
    print(f"Field names: {sorted(field_names)}")

    # Show sample records
    print("\n" + "="*60)
    print("SAMPLE RECORDS")
    print("="*60)
    for i, record in enumerate(sample_records):
        print(f"\n--- Record {i+1} ---")
        for key, value in record.items():
            # Truncate long values for display
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            print(f"{key}: {value_str}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify sleeper agents training data"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing data, don't download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=SLEEPER_DATA_PATH,
        help=f"Output path (default: {SLEEPER_DATA_PATH})"
    )
    args = parser.parse_args()

    if args.verify_only:
        print("Verify-only mode")
        success = verify_data(args.output)
    else:
        # Download if file doesn't exist or is too small
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output)
            if file_size > 1000000:  # > 1MB
                print(f"File already exists: {args.output} ({file_size / (1024*1024):.1f} MB)")
                print("Use --verify-only to check integrity, or delete to re-download")
                success = verify_data(args.output)
            else:
                print(f"File exists but too small ({file_size} bytes), re-downloading...")
                success = download_data(args.output) and verify_data(args.output)
        else:
            success = download_data(args.output) and verify_data(args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
