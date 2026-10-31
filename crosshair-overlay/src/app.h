#pragma once

#include <windows.h>

#include "settings.h"

// Services the main module provides to the settings dialog.
namespace app {

HINSTANCE Instance();

// Live settings. The dialog edits these directly and calls ApplyLive().
Settings& Current();
void ApplyLive();

// Draws the current crosshair centered in rc.
void PaintPreview(HDC dc, const RECT& rc);

// Saves the current settings and re-registers the hotkey.
// Returns false if the hotkey could not be registered (already taken).
bool Commit();

// While the settings dialog is open the global hotkey is released so the
// hotkey box can capture any combination.
void SuspendHotkey(bool suspend);

HICON Icon(bool large);

// Modal loops (menus, window dragging, message boxes, the color picker) block
// the main loop; while one runs a fallback timer keeps the crosshair moving.
void BeginModal();
void EndModal();

struct ModalScope {
    ModalScope() { BeginModal(); }
    ~ModalScope() { EndModal(); }
    ModalScope(const ModalScope&) = delete;
    ModalScope& operator=(const ModalScope&) = delete;
};

}  // namespace app
