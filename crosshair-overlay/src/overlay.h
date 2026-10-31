#pragma once

#include <windows.h>

#include <climits>

#include "renderer.h"
#include "settings.h"

// A click-through, always-on-top layered window that shows the crosshair at
// the mouse position or at the center of the game / active window.
class Overlay {
public:
    bool Create(HINSTANCE instance);
    void Destroy();

    // Re-renders the crosshair with new settings.
    void Apply(const Settings& s);

    void SetEnabled(bool enabled);
    bool Enabled() const { return enabled_; }

    // Polls the cursor and moves/shows/hides the window. Cheap; call often.
    // Returns true if the crosshair moved.
    bool Track();
    void ForceTopmost();

    // Draws the current crosshair centered at (x, y), e.g. for a preview.
    void PaintTo(HDC dc, int x, int y) const;

private:
    void UpdateLayered(POINT pos);
    POINT TargetPosition(POINT anchor) const;

    // Point the crosshair should be centered on; false = keep it hidden.
    bool Anchor(POINT& out);
    // Center of the foreground window's client area (or of its screen).
    POINT CenterPoint(POINT cursor, bool wholeScreen);
    // True when the foreground app looks like a game that took the mouse:
    // the cursor is hidden and the app is fullscreen or confines the cursor.
    // (Plain "hide pointer while typing" does not count.)
    bool GameHasMouse();
    // Foreground window, or nullptr if it is the desktop/taskbar or minimized.
    HWND AppWindow();

    HWND hwnd_ = nullptr;
    HDC memDC_ = nullptr;
    HBITMAP dib_ = nullptr;
    HGDIOBJ oldBitmap_ = nullptr;
    void* bits_ = nullptr;
    int dibSize_ = 0;

    CrosshairImage image_;
    BLENDFUNCTION blend_{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    int offsetX_ = 0, offsetY_ = 0;
    int positionMode_ = kPosAuto;

    HWND lastForeground_ = nullptr;   // cache for the desktop/taskbar check
    bool lastForegroundIsShell_ = true;

    bool enabled_ = true;
    bool shown_ = false;
    POINT lastPos_{LONG_MIN, LONG_MIN};
};
