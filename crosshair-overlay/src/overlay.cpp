#include "overlay.h"

#include <cstring>

namespace {

const wchar_t kOverlayClass[] = L"CrosshairOverlay.Overlay";

LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_NCHITTEST:
            return HTTRANSPARENT;          // never take mouse input
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;
        case WM_DPICHANGED:
            return 0;                      // keep the exact pixel size on every monitor
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace

bool Overlay::Create(HINSTANCE instance) {
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = OverlayProc;
    wc.hInstance = instance;
    wc.lpszClassName = kOverlayClass;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    // LAYERED + TRANSPARENT: per-pixel alpha and fully click-through.
    // TOOLWINDOW: no taskbar button and not in Alt+Tab. NOACTIVATE: never steals focus.
    hwnd_ = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW |
                                WS_EX_NOACTIVATE,
                            kOverlayClass, L"Crosshair Overlay", WS_POPUP, 0, 0, 1, 1, nullptr, nullptr,
                            instance, nullptr);
    if (!hwnd_) return false;

    HDC screen = GetDC(nullptr);
    memDC_ = CreateCompatibleDC(screen);
    ReleaseDC(nullptr, screen);
    return memDC_ != nullptr;
}

void Overlay::Destroy() {
    if (memDC_) {
        if (oldBitmap_) SelectObject(memDC_, oldBitmap_);
        DeleteDC(memDC_);
        memDC_ = nullptr;
    }
    if (dib_) {
        DeleteObject(dib_);
        dib_ = nullptr;
    }
    if (hwnd_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
}

void Overlay::Apply(const Settings& s) {
    RenderCrosshair(s, image_);
    offsetX_ = s.offsetX;
    offsetY_ = s.offsetY;
    hideWhenCursorHidden_ = s.hideWhenCursorHidden;
    blend_.SourceConstantAlpha = static_cast<BYTE>(s.opacity * 255 / 100);

    if (image_.size != dibSize_) {
        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
        bi.bmiHeader.biWidth = image_.size;
        bi.bmiHeader.biHeight = -image_.size;   // top-down
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        void* bits = nullptr;
        HBITMAP dib = CreateDIBSection(memDC_, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
        if (!dib) return;
        HGDIOBJ prev = SelectObject(memDC_, dib);
        if (!oldBitmap_) oldBitmap_ = prev;
        if (dib_) DeleteObject(dib_);
        dib_ = dib;
        bits_ = bits;
        dibSize_ = image_.size;
    }
    GdiFlush();
    std::memcpy(bits_, image_.pixels.data(), image_.pixels.size() * sizeof(uint32_t));

    POINT cursor{};
    GetCursorPos(&cursor);
    POINT pos = TargetPosition(cursor);
    UpdateLayered(pos);
    lastPos_ = pos;
    Track();
}

void Overlay::UpdateLayered(POINT pos) {
    if (!hwnd_ || !dib_) return;
    SIZE size{dibSize_, dibSize_};
    POINT src{0, 0};
    UpdateLayeredWindow(hwnd_, nullptr, &pos, &size, memDC_, &src, 0, &blend_, ULW_ALPHA);
}

POINT Overlay::TargetPosition(POINT cursor) const {
    return POINT{cursor.x - image_.center + offsetX_, cursor.y - image_.center + offsetY_};
}

bool Overlay::CursorHidden() const {
    CURSORINFO ci{sizeof(ci)};
    if (!GetCursorInfo(&ci)) return false;
    return (ci.flags & CURSOR_SHOWING) == 0;
}

void Overlay::SetEnabled(bool enabled) {
    enabled_ = enabled;
    Track();
}

bool Overlay::Track() {
    if (!hwnd_ || !dib_) return false;

    POINT cursor;
    const bool havePos = GetCursorPos(&cursor) != FALSE;   // fails e.g. on the secure desktop
    const bool want = enabled_ && havePos && !(hideWhenCursorHidden_ && CursorHidden());
    bool moved = false;

    if (want) {
        POINT pos = TargetPosition(cursor);
        if (pos.x != lastPos_.x || pos.y != lastPos_.y) {
            SetWindowPos(hwnd_, nullptr, pos.x, pos.y, 0, 0,
                         SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOSENDCHANGING);
            lastPos_ = pos;
            moved = true;
        }
        if (!shown_) {
            SetWindowPos(hwnd_, HWND_TOPMOST, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
            shown_ = true;
        }
    } else if (shown_) {
        ShowWindow(hwnd_, SW_HIDE);
        shown_ = false;
    }
    return moved;
}

void Overlay::ForceTopmost() {
    if (hwnd_ && shown_) {
        SetWindowPos(hwnd_, HWND_TOPMOST, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER);
    }
}

void Overlay::PaintTo(HDC dc, int x, int y) const {
    if (!dib_) return;
    AlphaBlend(dc, x - image_.center, y - image_.center, dibSize_, dibSize_, memDC_, 0, 0, dibSize_,
               dibSize_, blend_);
}
