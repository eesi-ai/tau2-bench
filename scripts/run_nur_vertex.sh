#!/usr/bin/env bash
# One bounded Nur voice diagnostic using Vertex for every simulator LLM call.
set -euo pipefail

if [[ -z "${EESI_API_KEY:-}" ]]; then
  echo "Set EESI_API_KEY for the Nur dev API." >&2
  exit 2
fi
if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT for Vertex AI." >&2
  exit 2
fi
export GOOGLE_CLOUD_LOCATION=global

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
cd "$repo_dir"
exec .venv/bin/tau2 run \
  --domain retail \
  --audio-native \
  --audio-native-provider eesi \
  --voice-synthesis-provider eesi \
  --user-llm vertex_gcloud/gemini-3.8-flash \
  --user-llm-args '{}' \
  --review-model vertex_gcloud/gemini-3.8-flash \
  --speech-complexity regular \
  --num-tasks 1 \
  --max-concurrency 1 \
  --max-steps-seconds 90 \
  "$@"
