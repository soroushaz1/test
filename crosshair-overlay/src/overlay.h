#pragma once

#include <windows.h>

#include <climits>

#include "renderer.h"
#include "settings.h"

// A click-through, always-on-top layered window that shows the crosshair and
// follows the mouse cursor.
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
    POINT TargetPosition(POINT cursor) const;
    bool CursorHidden() const;

    HWND hwnd_ = nullptr;
    HDC memDC_ = nullptr;
    HBITMAP dib_ = nullptr;
    HGDIOBJ oldBitmap_ = nullptr;
    void* bits_ = nullptr;
    int dibSize_ = 0;

    CrosshairImage image_;
    BLENDFUNCTION blend_{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    int offsetX_ = 0, offsetY_ = 0;
    bool hideWhenCursorHidden_ = false;

    bool enabled_ = true;
    bool shown_ = false;
    POINT lastPos_{LONG_MIN, LONG_MIN};
};
