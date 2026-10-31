// Crosshair Overlay - shows a customizable crosshair at the mouse position,
// above all other windows. Lives in the notification area (system tray).

#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <windowsx.h>
#include <mmsystem.h>

#include <string>

#include "app.h"
#include "overlay.h"
#include "resource.h"
#include "settings.h"
#include "settings_dialog.h"

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace {

constexpr UINT WM_APP_TRAY = WM_APP + 1;
constexpr UINT WM_APP_SHOW_SETTINGS = WM_APP + 2;
constexpr UINT kTrayId = 1;
constexpr int kHotkeyId = 1;
constexpr UINT_PTR kTimerTrack = 1;     // fallback while a modal loop (menu, drag) runs
constexpr UINT kTrackFallbackMs = 10;
constexpr UINT_PTR kTimerTopmost = 2;   // keep the overlay above other topmost windows
constexpr UINT_PTR kTimerTrayRetry = 3; // the taskbar may not exist yet at logon

enum MenuId : UINT { IDM_TOGGLE = 40001, IDM_SETTINGS, IDM_STARTUP, IDM_EXIT };

const wchar_t kMainClass[] = L"CrosshairOverlay.Main";
const wchar_t kMutexName[] = L"Local\\CrosshairOverlay.SingleInstance";
const wchar_t kAppName[] = L"Crosshair Overlay";

HINSTANCE g_instance = nullptr;
HWND g_main = nullptr;
Overlay g_overlay;
Settings g_settings;
UINT g_msgTaskbarCreated = 0;
bool g_trayAdded = false;
int g_trayRetries = 0;
bool g_hotkeySuspended = false;
int g_modalDepth = 0;
HANDLE g_pollTimer = nullptr;
bool g_pollTimerHighRes = false;
HICON g_iconOn = nullptr, g_iconOff = nullptr, g_iconLarge = nullptr;

HICON LoadAppIcon(int id, int size) {
    return static_cast<HICON>(LoadImageW(g_instance, MAKEINTRESOURCEW(id), IMAGE_ICON, size, size, LR_DEFAULTCOLOR));
}

// ---- Hotkey -----------------------------------------------------------------

bool RegisterToggleHotkey() {
    UnregisterHotKey(g_main, kHotkeyId);
    if (g_hotkeySuspended || !g_settings.hotkeyVk) return true;
    return RegisterHotKey(g_main, kHotkeyId, g_settings.hotkeyMods | MOD_NOREPEAT, g_settings.hotkeyVk) != FALSE;
}

// ---- Tray icon --------------------------------------------------------------

std::wstring TrayTip() {
    std::wstring tip = kAppName;
    tip += g_overlay.Enabled() ? L" - on" : L" - off";
    std::wstring hk = HotkeyToString(g_settings.hotkeyVk, g_settings.hotkeyMods);
    if (!hk.empty()) tip += L"\nToggle: " + hk;
    return tip;
}

void FillTrayData(NOTIFYICONDATAW& nid) {
    nid.cbSize = sizeof(nid);
    nid.hWnd = g_main;
    nid.uID = kTrayId;
    nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP;
    nid.uCallbackMessage = WM_APP_TRAY;
    nid.hIcon = g_overlay.Enabled() ? g_iconOn : g_iconOff;
    lstrcpynW(nid.szTip, TrayTip().c_str(), ARRAYSIZE(nid.szTip));
}

void AddTrayIcon() {
    NOTIFYICONDATAW nid{};
    FillTrayData(nid);
    g_trayAdded = Shell_NotifyIconW(NIM_ADD, &nid) != FALSE;
    if (g_trayAdded) {
        nid.uVersion = NOTIFYICON_VERSION_4;
        Shell_NotifyIconW(NIM_SETVERSION, &nid);
        KillTimer(g_main, kTimerTrayRetry);
    } else if (g_trayRetries++ < 60) {
        SetTimer(g_main, kTimerTrayRetry, 1000, nullptr);
    }
}

void UpdateTrayIcon() {
    if (!g_trayAdded) return;
    NOTIFYICONDATAW nid{};
    FillTrayData(nid);
    Shell_NotifyIconW(NIM_MODIFY, &nid);
}

void RemoveTrayIcon() {
    if (!g_trayAdded) return;
    NOTIFYICONDATAW nid{sizeof(nid)};
    nid.hWnd = g_main;
    nid.uID = kTrayId;
    Shell_NotifyIconW(NIM_DELETE, &nid);
    g_trayAdded = false;
}

void ShowBalloon(const std::wstring& title, const std::wstring& text) {
    if (!g_trayAdded) return;
    NOTIFYICONDATAW nid{sizeof(nid)};
    nid.hWnd = g_main;
    nid.uID = kTrayId;
    nid.uFlags = NIF_INFO;
    nid.dwInfoFlags = NIIF_INFO;
    lstrcpynW(nid.szInfoTitle, title.c_str(), ARRAYSIZE(nid.szInfoTitle));
    lstrcpynW(nid.szInfo, text.c_str(), ARRAYSIZE(nid.szInfo));
    Shell_NotifyIconW(NIM_MODIFY, &nid);
}

// ---- Cursor polling -----------------------------------------------------------

// A periodic waitable timer drives the cursor polling while the crosshair is
// on: every 1 ms while the mouse moves, every 15 ms once it has been still for
// a moment (saves power; the first movement is still caught within a frame).
// High-resolution waitable timers (Windows 10 1803+) are precise without
// changing the global timer resolution; older systems use timeBeginPeriod.
enum class PollRate { kOff, kFast, kIdle };
PollRate g_pollRate = PollRate::kOff;
DWORD g_lastMoveTick = 0;
constexpr LONG kFastPeriodMs = 1;
constexpr LONG kIdlePeriodMs = 15;
constexpr DWORD kIdleAfterMs = 500;

void CreatePollTimer() {
    g_pollTimer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    g_pollTimerHighRes = g_pollTimer != nullptr;
    if (!g_pollTimer) g_pollTimer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
}

void SetPollRate(PollRate rate) {
    if (rate == g_pollRate || !g_pollTimer) return;
    const bool wasOn = g_pollRate != PollRate::kOff;
    const bool on = rate != PollRate::kOff;
    if (on) {
        if (!wasOn && !g_pollTimerHighRes) timeBeginPeriod(1);
        const LONG period = (rate == PollRate::kFast) ? kFastPeriodMs : kIdlePeriodMs;
        LARGE_INTEGER due;
        due.QuadPart = -10000LL * period;   // 100 ns units, negative = relative
        SetWaitableTimer(g_pollTimer, &due, period, nullptr, nullptr, FALSE);
    } else {
        CancelWaitableTimer(g_pollTimer);
        if (wasOn && !g_pollTimerHighRes) timeEndPeriod(1);
    }
    g_pollRate = rate;
}

void UpdatePolling() {
    g_lastMoveTick = GetTickCount();
    SetPollRate(g_overlay.Enabled() ? PollRate::kFast : PollRate::kOff);
}

void PollCursor() {
    if (g_overlay.Track()) {
        g_lastMoveTick = GetTickCount();
        if (g_pollRate == PollRate::kIdle) SetPollRate(PollRate::kFast);
    } else if (g_pollRate == PollRate::kFast && GetTickCount() - g_lastMoveTick > kIdleAfterMs) {
        SetPollRate(PollRate::kIdle);
    }
}

void SetCrosshairEnabled(bool enabled) {
    g_overlay.SetEnabled(enabled);
    UpdatePolling();
    UpdateTrayIcon();
}

// ---- Tray menu ----------------------------------------------------------------

void ShowTrayMenu(POINT pt) {
    HMENU menu = CreatePopupMenu();
    std::wstring toggle = L"Show crosshair";
    std::wstring hk = HotkeyToString(g_settings.hotkeyVk, g_settings.hotkeyMods);
    if (!hk.empty()) toggle += L"\t" + hk;
    AppendMenuW(menu, MF_STRING | (g_overlay.Enabled() ? MF_CHECKED : 0), IDM_TOGGLE, toggle.c_str());
    AppendMenuW(menu, MF_STRING, IDM_SETTINGS, L"Settings...");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING | (IsStartupEnabled() ? MF_CHECKED : 0), IDM_STARTUP, L"Start with Windows");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING, IDM_EXIT, L"Exit");
    SetMenuDefaultItem(menu, IDM_SETTINGS, FALSE);

    // Required so the menu closes when the user clicks elsewhere.
    SetForegroundWindow(g_main);
    const UINT align = GetSystemMetrics(SM_MENUDROPALIGNMENT) ? TPM_RIGHTALIGN : TPM_LEFTALIGN;
    UINT cmd;
    {
        app::ModalScope modal;
        cmd = TrackPopupMenuEx(menu, TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON | TPM_BOTTOMALIGN | align, pt.x,
                               pt.y, g_main, nullptr);
    }
    PostMessageW(g_main, WM_NULL, 0, 0);
    DestroyMenu(menu);

    switch (cmd) {
        case IDM_TOGGLE:
            SetCrosshairEnabled(!g_overlay.Enabled());
            break;
        case IDM_SETTINGS:
            ShowSettingsDialog();
            break;
        case IDM_STARTUP:
            SetStartupEnabled(!IsStartupEnabled());
            break;
        case IDM_EXIT:
            DestroyWindow(g_main);
            break;
    }
}

// ---- Main (hidden) window -----------------------------------------------------

LRESULT CALLBACK MainProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CREATE:
            SetTimer(hwnd, kTimerTopmost, 500, nullptr);
            return 0;

        case WM_TIMER:
            if (wp == kTimerTrack) PollCursor();
            else if (wp == kTimerTopmost) g_overlay.ForceTopmost();
            else if (wp == kTimerTrayRetry) {
                KillTimer(hwnd, kTimerTrayRetry);
                AddTrayIcon();
            }
            return 0;

        case WM_HOTKEY:
            if (wp == kHotkeyId) SetCrosshairEnabled(!g_overlay.Enabled());
            return 0;

        case WM_APP_TRAY:
            switch (LOWORD(lp)) {
                case NIN_SELECT:
                case NIN_KEYSELECT:
                    ShowSettingsDialog();
                    break;
                case WM_CONTEXTMENU:
                    ShowTrayMenu(POINT{GET_X_LPARAM(wp), GET_Y_LPARAM(wp)});
                    break;
            }
            return 0;

        case WM_APP_SHOW_SETTINGS:
            ShowSettingsDialog();
            return 0;

        case WM_DISPLAYCHANGE:
            g_overlay.Track();
            return 0;

        case WM_DESTROY:
            CloseSettingsDialog();
            KillTimer(hwnd, kTimerTrack);
            KillTimer(hwnd, kTimerTopmost);
            KillTimer(hwnd, kTimerTrayRetry);
            UnregisterHotKey(hwnd, kHotkeyId);
            RemoveTrayIcon();
            PostQuitMessage(0);
            return 0;
    }
    if (msg == g_msgTaskbarCreated && g_msgTaskbarCreated) {   // Explorer restarted
        g_trayAdded = false;
        g_trayRetries = 0;
        AddTrayIcon();
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int RunMessageLoop() {
    MSG msg;
    for (;;) {
        const DWORD n = g_pollTimer ? 1 : 0;
        const DWORD r = MsgWaitForMultipleObjectsEx(n, &g_pollTimer, INFINITE, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
        if (n && r == WAIT_OBJECT_0) PollCursor();

        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return static_cast<int>(msg.wParam);
            HWND dlg = SettingsDialogHandle();
            if (dlg && IsDialogMessageW(dlg, &msg)) continue;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

// If another copy is already running, ask it to open its settings and quit.
bool ActivateExistingInstance() {
    HWND other = FindWindowW(kMainClass, nullptr);
    if (!other) return false;
    DWORD pid = 0;
    GetWindowThreadProcessId(other, &pid);
    AllowSetForegroundWindow(pid);
    PostMessageW(other, WM_APP_SHOW_SETTINGS, 0, 0);
    return true;
}

}  // namespace

// ---- Services for the settings dialog -------------------------------------

namespace app {

HINSTANCE Instance() { return g_instance; }
Settings& Current() { return g_settings; }
void ApplyLive() { g_overlay.Apply(g_settings); }

void PaintPreview(HDC dc, const RECT& rc) {
    g_overlay.PaintTo(dc, (rc.left + rc.right) / 2, (rc.top + rc.bottom) / 2);
}

bool Commit() {
    SaveSettings(g_settings);
    // Validate the hotkey even while it is suspended for the dialog.
    const bool wasSuspended = g_hotkeySuspended;
    g_hotkeySuspended = false;
    const bool ok = RegisterToggleHotkey();
    g_hotkeySuspended = wasSuspended;
    if (wasSuspended) UnregisterHotKey(g_main, kHotkeyId);
    UpdateTrayIcon();
    return ok;
}

void SuspendHotkey(bool suspend) {
    g_hotkeySuspended = suspend;
    RegisterToggleHotkey();
}

HICON Icon(bool large) { return large ? g_iconLarge : g_iconOn; }

void BeginModal() {
    if (g_modalDepth++ == 0) SetTimer(g_main, kTimerTrack, kTrackFallbackMs, nullptr);
}

void EndModal() {
    if (g_modalDepth > 0 && --g_modalDepth == 0) KillTimer(g_main, kTimerTrack);
}

}  // namespace app

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    g_instance = instance;

    HANDLE mutex = CreateMutexW(nullptr, TRUE, kMutexName);
    if (mutex && GetLastError() == ERROR_ALREADY_EXISTS) {
        ActivateExistingInstance();
        CloseHandle(mutex);
        return 0;
    }

    INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_STANDARD_CLASSES | ICC_BAR_CLASSES | ICC_HOTKEY_CLASS};
    InitCommonControlsEx(&icc);

    const int small = GetSystemMetrics(SM_CXSMICON);
    const int large = GetSystemMetrics(SM_CXICON);
    g_iconOn = LoadAppIcon(IDI_APP, small);
    g_iconOff = LoadAppIcon(IDI_APP_OFF, small);
    g_iconLarge = LoadAppIcon(IDI_APP, large);

    const bool firstRun = !LoadSettings(g_settings);
    if (firstRun) SaveSettings(g_settings);
    RefreshStartupPath();

    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = MainProc;
    wc.hInstance = instance;
    wc.lpszClassName = kMainClass;
    wc.hIcon = g_iconLarge;
    wc.hIconSm = g_iconOn;
    RegisterClassExW(&wc);

    g_msgTaskbarCreated = RegisterWindowMessageW(L"TaskbarCreated");
    // A hidden top-level window (not message-only) so it receives the
    // "TaskbarCreated" broadcast and can own the tray icon and hotkey.
    g_main = CreateWindowExW(0, kMainClass, kAppName, WS_OVERLAPPED, 0, 0, 0, 0, nullptr, nullptr, instance, nullptr);
    if (!g_main || !g_overlay.Create(instance)) {
        MessageBoxW(nullptr, L"Could not create the overlay window.", kAppName, MB_ICONERROR);
        return 1;
    }

    CreatePollTimer();
    g_overlay.Apply(g_settings);
    SetCrosshairEnabled(true);
    AddTrayIcon();

    const bool hotkeyOk = RegisterToggleHotkey();
    const std::wstring hk = HotkeyToString(g_settings.hotkeyVk, g_settings.hotkeyMods);
    if (!hotkeyOk) {
        ShowBalloon(kAppName, L"The hotkey " + hk + L" is used by another program. Open the settings to pick another one.");
    } else if (firstRun) {
        ShowBalloon(kAppName, L"The crosshair is running in the notification area. " +
                                  (hk.empty() ? std::wstring() : L"Press " + hk + L" to show/hide it. ") +
                                  L"Click the tray icon for settings.");
    }
    if (firstRun) ShowSettingsDialog();

    const int code = RunMessageLoop();

    g_overlay.SetEnabled(false);
    UpdatePolling();   // stops the timer (and timeEndPeriod on old systems)
    g_overlay.Destroy();
    if (g_pollTimer) CloseHandle(g_pollTimer);
    if (mutex) CloseHandle(mutex);
    return code;
}
