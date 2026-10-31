#include "settings_dialog.h"

#include <commctrl.h>
#include <commdlg.h>

#include <algorithm>
#include <iterator>

#include "app.h"
#include "resource.h"
#include "settings.h"

namespace {

using namespace limits;

struct SliderRow {
    int base;                       // label id; trackbar = base + 1, edit = base + 2
    int Settings::*field;
    int min, max;
    bool Settings::*enabledBy;      // row is disabled while this option is off
};

const SliderRow kSliders[] = {
    {IDC_LENGTH, &Settings::length, kLengthMin, kLengthMax, &Settings::showCross},
    {IDC_THICK, &Settings::thickness, kThicknessMin, kThicknessMax, &Settings::showCross},
    {IDC_GAP, &Settings::gap, kGapMin, kGapMax, &Settings::showCross},
    {IDC_ROT, &Settings::rotation, kRotationMin, kRotationMax, &Settings::showCross},
    {IDC_DOT, &Settings::dotSize, kDotMin, kDotMax, &Settings::showDot},
    {IDC_RADIUS, &Settings::circleRadius, kRadiusMin, kRadiusMax, &Settings::showCircle},
    {IDC_CTHICK, &Settings::circleThickness, kCircleThickMin, kCircleThickMax, &Settings::showCircle},
    {IDC_OPACITY, &Settings::opacity, kOpacityMin, kOpacityMax, nullptr},
    {IDC_OTHICK, &Settings::outlineThickness, kOutlineMin, kOutlineMax, &Settings::outline},
    {IDC_OFFX, &Settings::offsetX, kOffsetMin, kOffsetMax, nullptr},
    {IDC_OFFY, &Settings::offsetY, kOffsetMin, kOffsetMax, nullptr},
};

struct CheckRow {
    int id;
    bool Settings::*field;
    bool Settings::*enabledBy;
};

const CheckRow kChecks[] = {
    {IDC_SHOW_CROSS, &Settings::showCross, nullptr},
    {IDC_TSTYLE, &Settings::tStyle, &Settings::showCross},
    {IDC_SHOW_DOT, &Settings::showDot, nullptr},
    {IDC_DOT_ROUND, &Settings::dotRound, &Settings::showDot},
    {IDC_SHOW_CIRCLE, &Settings::showCircle, nullptr},
    {IDC_OUTLINE, &Settings::outline, nullptr},
};

// Order must match the PositionMode enum.
const wchar_t* const kPositionNames[] = {
    L"Auto: mouse, or center in games",
    L"Always follow the mouse",
    L"Mouse, hide when cursor is hidden",
    L"Center of the active window",
    L"Center of the screen",
};
static_assert(std::size(kPositionNames) == kPosCount, "one name per PositionMode");

// ---- Presets --------------------------------------------------------------

// Copies only the look of the crosshair (not offsets, hotkey or behavior).
void CopyAppearance(Settings& dst, const Settings& src) {
    dst.showCross = src.showCross;
    dst.length = src.length;
    dst.thickness = src.thickness;
    dst.gap = src.gap;
    dst.rotation = src.rotation;
    dst.tStyle = src.tStyle;
    dst.showDot = src.showDot;
    dst.dotSize = src.dotSize;
    dst.dotRound = src.dotRound;
    dst.showCircle = src.showCircle;
    dst.circleRadius = src.circleRadius;
    dst.circleThickness = src.circleThickness;
    dst.color = src.color;
    dst.opacity = src.opacity;
    dst.outline = src.outline;
    dst.outlineColor = src.outlineColor;
    dst.outlineThickness = src.outlineThickness;
}

const wchar_t* const kPresetNames[] = {
    L"Default (green cross)", L"Dot", L"Circle + dot", L"X (diagonal)",
    L"T-style", L"Long thin lines", L"Solid plus",
};

Settings PresetAppearance(int index) {
    Settings p;   // defaults
    switch (index) {
        case 1:
            p.showCross = false;
            p.showDot = true;
            p.dotSize = 4;
            p.dotRound = true;
            p.color = RGB(255, 40, 40);
            break;
        case 2:
            p.showCross = false;
            p.showCircle = true;
            p.circleRadius = 12;
            p.circleThickness = 2;
            p.showDot = true;
            p.dotSize = 3;
            p.dotRound = true;
            p.color = RGB(0, 230, 255);
            break;
        case 3:
            p.rotation = 45;
            p.length = 10;
            p.gap = 5;
            p.color = RGB(255, 230, 0);
            break;
        case 4:
            p.tStyle = true;
            p.length = 9;
            p.gap = 3;
            p.showDot = true;
            p.dotSize = 2;
            p.color = RGB(255, 0, 255);
            break;
        case 5:
            p.length = 80;
            p.thickness = 1;
            p.gap = 10;
            p.color = RGB(255, 255, 255);
            break;
        case 6:
            p.length = 10;
            p.thickness = 3;
            p.gap = 0;
            p.color = RGB(0, 255, 0);
            break;
    }
    return p;
}

// ---- Dialog state ---------------------------------------------------------

HWND g_dlg = nullptr;
Settings g_original;            // restored on Cancel
bool g_startupOriginal = false;
bool g_loading = false;         // suppresses feedback while filling controls
HBITMAP g_swatch[2] = {};       // color button images
COLORREF g_customColors[16] = {};

const SliderRow* FindSlider(int ctrlId) {
    for (const SliderRow& r : kSliders)
        if (ctrlId >= r.base && ctrlId <= r.base + 2) return &r;
    return nullptr;
}

const CheckRow* FindCheck(int ctrlId) {
    for (const CheckRow& c : kChecks)
        if (c.id == ctrlId) return &c;
    return nullptr;
}

BYTE ModsToHotkeyFlags(UINT mods) {
    BYTE f = 0;
    if (mods & MOD_SHIFT) f |= HOTKEYF_SHIFT;
    if (mods & MOD_CONTROL) f |= HOTKEYF_CONTROL;
    if (mods & MOD_ALT) f |= HOTKEYF_ALT;
    return f;
}

UINT HotkeyFlagsToMods(BYTE f) {
    UINT m = 0;
    if (f & HOTKEYF_SHIFT) m |= MOD_SHIFT;
    if (f & HOTKEYF_CONTROL) m |= MOD_CONTROL;
    if (f & HOTKEYF_ALT) m |= MOD_ALT;
    return m;
}

void UpdateSwatch(HWND dlg, int btnId, int slot, COLORREF color) {
    HWND btn = GetDlgItem(dlg, btnId);
    RECT rc;
    GetClientRect(btn, &rc);
    const int w = std::max(8, static_cast<int>(rc.right) - 14);
    const int h = std::max(6, static_cast<int>(rc.bottom) - 10);

    HDC screen = GetDC(nullptr);
    HDC mem = CreateCompatibleDC(screen);
    HBITMAP bmp = CreateCompatibleBitmap(screen, w, h);
    ReleaseDC(nullptr, screen);
    HGDIOBJ old = SelectObject(mem, bmp);
    RECT r{0, 0, w, h};
    HBRUSH border = CreateSolidBrush(RGB(90, 90, 90));
    FillRect(mem, &r, border);
    DeleteObject(border);
    InflateRect(&r, -1, -1);
    HBRUSH fill = CreateSolidBrush(color);
    FillRect(mem, &r, fill);
    DeleteObject(fill);
    SelectObject(mem, old);
    DeleteDC(mem);

    HBITMAP prev = reinterpret_cast<HBITMAP>(
        SendMessageW(btn, BM_SETIMAGE, IMAGE_BITMAP, reinterpret_cast<LPARAM>(bmp)));
    if (prev && prev != g_swatch[slot]) DeleteObject(prev);
    if (g_swatch[slot]) DeleteObject(g_swatch[slot]);
    g_swatch[slot] = bmp;
}

void UpdateEnabledState(HWND dlg) {
    const Settings& s = app::Current();
    for (const SliderRow& r : kSliders) {
        const BOOL on = !r.enabledBy || s.*(r.enabledBy);
        for (int i = 0; i < 3; ++i) EnableWindow(GetDlgItem(dlg, r.base + i), on);
    }
    for (const CheckRow& c : kChecks) {
        if (c.enabledBy) EnableWindow(GetDlgItem(dlg, c.id), s.*(c.enabledBy));
    }
    EnableWindow(GetDlgItem(dlg, IDC_OUTLINE_COLOR_BTN), s.outline);
}

void RefreshPreview(HWND dlg) { InvalidateRect(GetDlgItem(dlg, IDC_PREVIEW), nullptr, FALSE); }

void LoadControls(HWND dlg) {
    const Settings& s = app::Current();
    g_loading = true;
    for (const SliderRow& r : kSliders) {
        SendDlgItemMessageW(dlg, r.base + 1, TBM_SETPOS, TRUE, s.*(r.field));
        SetDlgItemInt(dlg, r.base + 2, static_cast<UINT>(s.*(r.field)), TRUE);
    }
    for (const CheckRow& c : kChecks) {
        CheckDlgButton(dlg, c.id, s.*(c.field) ? BST_CHECKED : BST_UNCHECKED);
    }
    SendDlgItemMessageW(dlg, IDC_HOTKEY, HKM_SETHOTKEY,
                        MAKEWORD(s.hotkeyVk, ModsToHotkeyFlags(s.hotkeyMods)), 0);
    SendDlgItemMessageW(dlg, IDC_POSMODE, CB_SETCURSEL, s.positionMode, 0);
    UpdateSwatch(dlg, IDC_COLOR_BTN, 0, s.color);
    UpdateSwatch(dlg, IDC_OUTLINE_COLOR_BTN, 1, s.outlineColor);
    UpdateEnabledState(dlg);
    g_loading = false;
}

void Changed(HWND dlg) {
    app::Current().Clamp();
    app::ApplyLive();
    UpdateEnabledState(dlg);
    RefreshPreview(dlg);
}

void InitControls(HWND dlg) {
    for (const SliderRow& r : kSliders) {
        HWND tb = GetDlgItem(dlg, r.base + 1);
        SendMessageW(tb, TBM_SETRANGEMIN, FALSE, r.min);
        SendMessageW(tb, TBM_SETRANGEMAX, TRUE, r.max);
        SendMessageW(tb, TBM_SETLINESIZE, 0, 1);
        SendMessageW(tb, TBM_SETPAGESIZE, 0, std::max(1, (r.max - r.min) / 20));
        SendDlgItemMessageW(dlg, r.base + 2, EM_SETLIMITTEXT, 4, 0);
    }

    HWND combo = GetDlgItem(dlg, IDC_PRESET);
    for (const wchar_t* name : kPresetNames)
        SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(name));
    SendMessageW(combo, CB_SETCUEBANNER, 0, reinterpret_cast<LPARAM>(L"Choose a preset..."));

    HWND posCombo = GetDlgItem(dlg, IDC_POSMODE);
    for (const wchar_t* name : kPositionNames)
        SendMessageW(posCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(name));

    // Plain keys or Shift+key would hijack normal typing; turn them into Ctrl+Alt+key.
    SendDlgItemMessageW(dlg, IDC_HOTKEY, HKM_SETRULES, HKCOMB_NONE | HKCOMB_S,
                        MAKELPARAM(HOTKEYF_CONTROL | HOTKEYF_ALT, 0));

    g_startupOriginal = IsStartupEnabled();
    CheckDlgButton(dlg, IDC_STARTUP, g_startupOriginal ? BST_CHECKED : BST_UNCHECKED);

    SendMessageW(dlg, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(app::Icon(true)));
    SendMessageW(dlg, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(app::Icon(false)));
}

void OnSliderScrolled(HWND dlg, HWND trackbar) {
    const SliderRow* r = FindSlider(GetDlgCtrlID(trackbar));
    if (!r || g_loading) return;
    const int v = static_cast<int>(SendMessageW(trackbar, TBM_GETPOS, 0, 0));
    app::Current().*(r->field) = v;
    g_loading = true;
    SetDlgItemInt(dlg, r->base + 2, static_cast<UINT>(v), TRUE);
    g_loading = false;
    Changed(dlg);
}

void OnEditChanged(HWND dlg, const SliderRow& r) {
    if (g_loading) return;
    BOOL ok = FALSE;
    const int v = static_cast<int>(GetDlgItemInt(dlg, r.base + 2, &ok, TRUE));
    if (!ok) return;   // empty or just "-" while typing
    const int clamped = std::clamp(v, r.min, r.max);
    app::Current().*(r.field) = clamped;
    SendDlgItemMessageW(dlg, r.base + 1, TBM_SETPOS, TRUE, clamped);
    Changed(dlg);
}

void OnEditKillFocus(HWND dlg, const SliderRow& r) {
    // Show the value that is actually in use (e.g. after clamping).
    g_loading = true;
    SetDlgItemInt(dlg, r.base + 2, static_cast<UINT>(app::Current().*(r.field)), TRUE);
    g_loading = false;
}

void PickColor(HWND dlg, COLORREF& color, int btnId, int slot) {
    CHOOSECOLORW cc{sizeof(cc)};
    cc.hwndOwner = dlg;
    cc.rgbResult = color;
    cc.lpCustColors = g_customColors;
    cc.Flags = CC_FULLOPEN | CC_RGBINIT | CC_ANYCOLOR;
    BOOL picked;
    {
        app::ModalScope modal;
        picked = ChooseColorW(&cc);
    }
    if (picked) {
        color = cc.rgbResult;
        UpdateSwatch(dlg, btnId, slot, color);
        Changed(dlg);
    }
}

void ApplyPreset(HWND dlg, int index) {
    if (index < 0 || index >= static_cast<int>(std::size(kPresetNames))) return;
    CopyAppearance(app::Current(), PresetAppearance(index));
    LoadControls(dlg);
    Changed(dlg);
}

void DrawPreview(const DRAWITEMSTRUCT* di) {
    const RECT& rc = di->rcItem;
    const int w = rc.right - rc.left, h = rc.bottom - rc.top;
    HDC mem = CreateCompatibleDC(di->hDC);
    HBITMAP bmp = CreateCompatibleBitmap(di->hDC, w, h);
    HGDIOBJ old = SelectObject(mem, bmp);

    // Dark and light halves so the crosshair can be judged on both.
    RECT left{0, 0, w / 2, h}, right{w / 2, 0, w, h}, all{0, 0, w, h};
    HBRUSH dark = CreateSolidBrush(RGB(36, 38, 42));
    HBRUSH light = CreateSolidBrush(RGB(226, 229, 233));
    HBRUSH frame = CreateSolidBrush(RGB(120, 120, 120));
    FillRect(mem, &left, dark);
    FillRect(mem, &right, light);
    app::PaintPreview(mem, all);
    FrameRect(mem, &all, frame);
    DeleteObject(dark);
    DeleteObject(light);
    DeleteObject(frame);

    BitBlt(di->hDC, rc.left, rc.top, w, h, mem, 0, 0, SRCCOPY);
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteDC(mem);
}

void OnOk(HWND dlg) {
    Settings& s = app::Current();
    const WORD hk = static_cast<WORD>(SendDlgItemMessageW(dlg, IDC_HOTKEY, HKM_GETHOTKEY, 0, 0));
    s.hotkeyVk = LOBYTE(hk);
    s.hotkeyMods = s.hotkeyVk ? HotkeyFlagsToMods(HIBYTE(hk)) : 0;

    const bool startup = IsDlgButtonChecked(dlg, IDC_STARTUP) == BST_CHECKED;
    if (startup != g_startupOriginal) SetStartupEnabled(startup);

    if (!app::Commit()) {
        std::wstring msg = L"The hotkey " + HotkeyToString(s.hotkeyVk, s.hotkeyMods) +
                           L" is already used by another program.\n\nYour other settings were saved. "
                           L"Please pick a different hotkey.";
        {
            app::ModalScope modal;
            MessageBoxW(dlg, msg.c_str(), L"Crosshair Overlay", MB_OK | MB_ICONWARNING);
        }
        g_original = s;
        g_startupOriginal = startup;
        SetFocus(GetDlgItem(dlg, IDC_HOTKEY));
        return;
    }
    DestroyWindow(dlg);
}

void OnCancel(HWND dlg) {
    app::Current() = g_original;
    app::ApplyLive();
    DestroyWindow(dlg);
}

INT_PTR CALLBACK DialogProc(HWND dlg, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_INITDIALOG:
            g_dlg = dlg;
            g_original = app::Current();
            app::SuspendHotkey(true);
            InitControls(dlg);
            LoadControls(dlg);
            return TRUE;

        case WM_HSCROLL:
            if (lp) OnSliderScrolled(dlg, reinterpret_cast<HWND>(lp));
            return TRUE;

        case WM_COMMAND: {
            const int id = LOWORD(wp);
            const int code = HIWORD(wp);
            if (const SliderRow* r = FindSlider(id); r && id == r->base + 2) {
                if (code == EN_CHANGE) OnEditChanged(dlg, *r);
                else if (code == EN_KILLFOCUS) OnEditKillFocus(dlg, *r);
                return TRUE;
            }
            if (const CheckRow* c = FindCheck(id); c && code == BN_CLICKED) {
                app::Current().*(c->field) = IsDlgButtonChecked(dlg, id) == BST_CHECKED;
                Changed(dlg);
                return TRUE;
            }
            switch (id) {
                case IDC_COLOR_BTN:
                    PickColor(dlg, app::Current().color, IDC_COLOR_BTN, 0);
                    return TRUE;
                case IDC_OUTLINE_COLOR_BTN:
                    PickColor(dlg, app::Current().outlineColor, IDC_OUTLINE_COLOR_BTN, 1);
                    return TRUE;
                case IDC_POSMODE:
                    if (code == CBN_SELCHANGE && !g_loading) {
                        const int sel = static_cast<int>(SendDlgItemMessageW(dlg, IDC_POSMODE, CB_GETCURSEL, 0, 0));
                        if (sel >= 0) {
                            app::Current().positionMode = sel;
                            Changed(dlg);
                        }
                    }
                    return TRUE;
                case IDC_PRESET:
                    if (code == CBN_SELCHANGE)
                        ApplyPreset(dlg, static_cast<int>(SendDlgItemMessageW(dlg, IDC_PRESET, CB_GETCURSEL, 0, 0)));
                    return TRUE;
                case IDOK:
                    OnOk(dlg);
                    return TRUE;
                case IDCANCEL:
                    OnCancel(dlg);
                    return TRUE;
            }
            break;
        }

        case WM_DRAWITEM: {
            auto* di = reinterpret_cast<const DRAWITEMSTRUCT*>(lp);
            if (di->CtlID == IDC_PREVIEW) {
                DrawPreview(di);
                return TRUE;
            }
            break;
        }

        // Dragging the window or opening its system menu runs a modal loop.
        case WM_ENTERSIZEMOVE:
        case WM_ENTERMENULOOP:
            app::BeginModal();
            return TRUE;
        case WM_EXITSIZEMOVE:
        case WM_EXITMENULOOP:
            app::EndModal();
            return TRUE;

        case WM_CLOSE:
            OnCancel(dlg);
            return TRUE;

        case WM_DESTROY:
            for (HBITMAP& b : g_swatch) {
                if (b) DeleteObject(b);
                b = nullptr;
            }
            g_dlg = nullptr;
            app::SuspendHotkey(false);
            return TRUE;
    }
    return FALSE;
}

}  // namespace

void ShowSettingsDialog() {
    if (!g_dlg) {
        CreateDialogParamW(app::Instance(), MAKEINTRESOURCEW(IDD_SETTINGS), nullptr, DialogProc, 0);
        if (!g_dlg) return;
        ShowWindow(g_dlg, SW_SHOW);
    } else if (IsIconic(g_dlg)) {
        ShowWindow(g_dlg, SW_RESTORE);
    }
    SetForegroundWindow(g_dlg);
}

HWND SettingsDialogHandle() { return g_dlg; }

void CloseSettingsDialog() {
    if (g_dlg) OnCancel(g_dlg);
}
