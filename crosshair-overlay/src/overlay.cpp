#include "overlay.h"

#include <cstring>
#include <cwchar>

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

bool IsCursorHidden() {
    CURSORINFO ci{sizeof(ci)};
    if (!GetCursorInfo(&ci)) return false;
    // Games hide the pointer with ShowCursor(FALSE) or SetCursor(NULL).
    return (ci.flags & CURSOR_SHOWING) == 0 || ci.hCursor == nullptr;
}

bool IsShellWindow(HWND hwnd) {
    wchar_t cls[64];
    if (!GetClassNameW(hwnd, cls, ARRAYSIZE(cls))) return true;
    return wcscmp(cls, L"Progman") == 0 || wcscmp(cls, L"WorkerW") == 0 ||
           wcscmp(cls, L"Shell_TrayWnd") == 0 || wcscmp(cls, L"Shell_SecondaryTrayWnd") == 0;
}

RECT MonitorRect(HMONITOR monitor) {
    MONITORINFO mi{sizeof(mi)};
    GetMonitorInfoW(monitor, &mi);
    return mi.rcMonitor;
}

POINT RectCenter(const RECT& r) { return POINT{r.left + (r.right - r.left) / 2, r.top + (r.bottom - r.top) / 2}; }

bool IsCursorClipped() {
    RECT clip;
    if (!GetClipCursor(&clip)) return false;
    const int vx = GetSystemMetrics(SM_XVIRTUALSCREEN), vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
    const int vw = GetSystemMetrics(SM_CXVIRTUALSCREEN), vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    return clip.left > vx || clip.top > vy || clip.right < vx + vw || clip.bottom < vy + vh;
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
    positionMode_ = s.positionMode;
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

    POINT anchor{};
    if (!Anchor(anchor)) GetCursorPos(&anchor);
    POINT pos = TargetPosition(anchor);
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

POINT Overlay::TargetPosition(POINT anchor) const {
    return POINT{anchor.x - image_.center + offsetX_, anchor.y - image_.center + offsetY_};
}

HWND Overlay::AppWindow() {
    HWND fg = GetForegroundWindow();
    if (fg != lastForeground_) {
        lastForeground_ = fg;
        lastForegroundIsShell_ = !fg || IsShellWindow(fg);
    }
    if (lastForegroundIsShell_ || IsIconic(fg)) return nullptr;
    return fg;
}

POINT Overlay::CenterPoint(POINT cursor, bool wholeScreen) {
    HWND app = AppWindow();
    if (!app) return RectCenter(MonitorRect(MonitorFromPoint(cursor, MONITOR_DEFAULTTONEAREST)));
    if (!wholeScreen) {
        RECT rc;
        if (GetClientRect(app, &rc) && rc.right > 0 && rc.bottom > 0) {
            POINT p{rc.right / 2, rc.bottom / 2};
            if (ClientToScreen(app, &p)) return p;
        }
    }
    return RectCenter(MonitorRect(MonitorFromWindow(app, MONITOR_DEFAULTTONEAREST)));
}

bool Overlay::GameHasMouse() {
    if (!IsCursorHidden()) return false;
    HWND app = AppWindow();
    if (!app) return false;
    if (IsCursorClipped()) return true;
    // Fullscreen or borderless-fullscreen: the window covers its whole monitor.
    RECT wr;
    if (!GetWindowRect(app, &wr)) return false;
    const RECT mr = MonitorRect(MonitorFromWindow(app, MONITOR_DEFAULTTONEAREST));
    return wr.left <= mr.left && wr.top <= mr.top && wr.right >= mr.right && wr.bottom >= mr.bottom;
}

bool Overlay::Anchor(POINT& out) {
    POINT cursor;
    if (!GetCursorPos(&cursor)) return false;   // fails e.g. on the secure desktop
    switch (positionMode_) {
        case kPosMouse:
            out = cursor;
            return true;
        case kPosMouseHide:
            out = cursor;
            return !IsCursorHidden();
        case kPosWindowCenter:
            out = CenterPoint(cursor, false);
            return true;
        case kPosScreenCenter:
            out = CenterPoint(cursor, true);
            return true;
        case kPosAuto:
        default:
            out = GameHasMouse() ? CenterPoint(cursor, false) : cursor;
            return true;
    }
}

void Overlay::SetEnabled(bool enabled) {
    enabled_ = enabled;
    Track();
}

bool Overlay::Track() {
    if (!hwnd_ || !dib_) return false;

    POINT anchor{};
    const bool want = enabled_ && Anchor(anchor);
    bool moved = false;

    if (want) {
        POINT pos = TargetPosition(anchor);
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
