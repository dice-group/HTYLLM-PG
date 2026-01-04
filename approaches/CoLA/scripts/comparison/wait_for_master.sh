#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?missing host}"
PORT="${2:?missing port}"
TIMEOUT="${3:-90}"

deadline=$((SECONDS + TIMEOUT))
while (( SECONDS < deadline )); do
  if (echo >"/dev/tcp/${HOST}/${PORT}") >/dev/null 2>&1; then
    echo "[INFO] Master reachable, proceeding."
    exit 0
  fi
  sleep 1
done

echo "[ERROR] Master not reachable within timeout." >&2
exit 1
