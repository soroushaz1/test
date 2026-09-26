#!/usr/bin/env bash
# Install and wire up caveman (terse-output skill) and rtk (command-output
# compressor) for Claude Code on Linux/macOS.
set -euo pipefail

command -v claude >/dev/null || { echo "claude CLI not found; install Claude Code first" >&2; exit 1; }

# --- rtk ---------------------------------------------------------------
if ! command -v rtk >/dev/null; then
  if command -v brew >/dev/null; then
    brew install rtk
  else
    curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    grep -qs '.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  fi
fi
rtk --version
rtk init -g   # registers the global Claude Code hook

# --- caveman -----------------------------------------------------------
claude plugin marketplace add JuliusBrussee/caveman || true   # no-op if already added
claude plugin install caveman@caveman

echo "Done. Restart Claude Code, then run /caveman inside it and 'rtk gain' to see savings."
