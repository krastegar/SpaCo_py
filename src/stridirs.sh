#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# StripDirs Script
# -----------------------------------------------------------------------------
# This script, named "stripdirs", is designed to process R-script files by
# removing their directory paths, leaving only the base script name with the
# ".RDS" extension. The script accepts multiple R-script files as input and
# processes each one individually, ensuring that only the base script name
# remains in the output.
#
# USAGE:
#   ./stripdirs [OPTIONS] <file1.R> [file2.R ...]
#
# OPTIONS:
#   -h    Display this help message and exit.
#   -v    Display the script version and exit.
#
# REQUIREMENTS:
#   - The script requires at least one R-script file as an argument.
#   - Each specified file must exist and be readable.
# -----------------------------------------------------------------------------

set -euo pipefail

VERSION="1.3.0"
SCRIPT_NAME=$(basename "$0")
TARGET_DIR=""

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] <file1> [file2 ...]
Strip all directory components from any file paths (with extensions)
and prefix them with the provided target directory.

Options:
  -d DIR, --dir DIR   Target directory to prefix (required)
  -h, --help          Show this help message and exit
  -v, --version       Show version and exit
EOF
}

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)
      if [[ -n "${2-}" && ! "$2" =~ ^- ]]; then
        TARGET_DIR="$2"
        shift 2
      else
        echo "Error: --dir requires a non-empty argument." >&2
        usage
        exit 1
      fi
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -v|--version)
      echo "$SCRIPT_NAME version $VERSION"
      exit 0
      ;;
    --) # end of options
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *) # no more options
      break
      ;;
  esac
done

# Ensure target directory was provided
if [[ -z "$TARGET_DIR" ]]; then
  echo "Error: Target directory must be specified with -d or --dir." >&2
  usage
  exit 1
fi

# Make sure TARGET_DIR does not end with a slash
TARGET_DIR="${TARGET_DIR%/}"

# Require at least one file to process
if [[ $# -lt 1 ]]; then
  echo "Error: At least one file must be specified." >&2
  usage
  exit 1
fi

# Process each file
for file in "$@"; do
  if [[ ! -f "$file" ]]; then
    echo "Warning: '$file' not found, skipping." >&2
    continue
  fi

  # Strip any /dir/.../prefix and prefix with TARGET_DIR
  sed -r -i \
    -e "s@(/\S+/)+([^/]+\.[^/]+)@${TARGET_DIR}/\2@g" \
    "$file"

  echo "Processed: $file"
done









