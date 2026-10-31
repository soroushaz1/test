#!/usr/bin/env bash
# Cross-compiles CrosshairOverlay.exe on Linux/macOS/MSYS2 with MinGW-w64.
#   Ubuntu/Debian:  sudo apt install g++-mingw-w64-x86-64
#   MSYS2 (Windows): pacman -S mingw-w64-x86_64-gcc   (then run with CXX=g++ WINDRES=windres)
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-x86_64-w64-mingw32-g++}"
WINDRES="${WINDRES:-x86_64-w64-mingw32-windres}"
OUT_DIR="${OUT_DIR:-build}"
mkdir -p "$OUT_DIR"

DEFS="-DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00"

"$WINDRES" --input-format=rc --output-format=coff -I src -I res src/app.rc -o "$OUT_DIR/app.res.o"

"$CXX" -std=c++17 -O2 -Wall -Wextra -Wno-missing-field-initializers $DEFS \
    src/main.cpp src/overlay.cpp src/renderer.cpp src/settings.cpp src/settings_dialog.cpp \
    "$OUT_DIR/app.res.o" \
    -o "$OUT_DIR/CrosshairOverlay.exe" \
    -municode -mwindows -static -static-libgcc -static-libstdc++ -s \
    -lcomctl32 -lcomdlg32 -lshell32 -lgdi32 -luser32 -ladvapi32 -lmsimg32 -lwinmm

echo "Built $OUT_DIR/CrosshairOverlay.exe"
