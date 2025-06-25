#!/bin/bash

# Usage check
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <directory_path> <include_regex> [exclude_regex]"
  exit 1
fi

DIR="$1"
INCLUDE_REGEX="$2"
EXCLUDE_REGEX="$3"

# Check if directory exists
if [ ! -d "$DIR" ]; then
  echo "Error: Directory '$DIR' does not exist."
  exit 1
fi

echo "Searching for pattern: '$INCLUDE_REGEX'"
[ -n "$EXCLUDE_REGEX" ] && echo "Excluding matches that fit: '$EXCLUDE_REGEX'"

# Recursive file search
find "$DIR" -type f | while read -r FILE; do
  MATCHES=$(grep -En "$INCLUDE_REGEX" "$FILE")

  if [ -n "$EXCLUDE_REGEX" ]; then
    MATCHES=$(echo "$MATCHES" | grep -Ev "$EXCLUDE_REGEX")
  fi

  if [ -n "$MATCHES" ]; then
    echo "Matches in file: $FILE"
    echo "$MATCHES"
    echo "---"
  fi
done
