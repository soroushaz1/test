#pragma once

#include <windows.h>
#include <string>

// All user-adjustable options. Sizes are in physical screen pixels.
struct Settings {
    // Cross lines
    bool showCross = true;
    int length = 8;          // length of each arm
    int thickness = 2;       // line thickness
    int gap = 4;             // empty space between the center and each arm
    int rotation = 0;        // degrees, 45 gives an "X"
    bool tStyle = false;     // hide the top arm

    // Center dot
    bool showDot = false;
    int dotSize = 2;
    bool dotRound = false;

    // Circle
    bool showCircle = false;
    int circleRadius = 14;
    int circleThickness = 1;

    // Colors
    COLORREF color = RGB(0, 255, 0);
    int opacity = 100;       // percent
    bool outline = true;
    COLORREF outlineColor = RGB(0, 0, 0);
    int outlineThickness = 1;

    // Position relative to the mouse hotspot
    int offsetX = 0;
    int offsetY = 0;

    // Behavior
    bool hideWhenCursorHidden = false;
    UINT hotkeyVk = 'X';
    UINT hotkeyMods = MOD_CONTROL | MOD_ALT;   // MOD_* flags

    void Clamp();
};

// Allowed ranges, shared by the settings loader and the settings dialog.
namespace limits {
constexpr int kLengthMin = 0, kLengthMax = 500;
constexpr int kThicknessMin = 1, kThicknessMax = 40;
constexpr int kGapMin = 0, kGapMax = 200;
constexpr int kRotationMin = 0, kRotationMax = 359;
constexpr int kDotMin = 1, kDotMax = 60;
constexpr int kRadiusMin = 1, kRadiusMax = 400;
constexpr int kCircleThickMin = 1, kCircleThickMax = 40;
constexpr int kOpacityMin = 5, kOpacityMax = 100;
constexpr int kOutlineMin = 1, kOutlineMax = 10;
constexpr int kOffsetMin = -500, kOffsetMax = 500;
}  // namespace limits

// Returns the path of the settings file. A settings.ini next to the exe
// (portable mode) wins; otherwise %APPDATA%\CrosshairOverlay\settings.ini.
std::wstring SettingsFilePath();

// Loads settings; returns false if no settings file existed yet.
bool LoadSettings(Settings& s);
bool SaveSettings(const Settings& s);

// "Run at login" via HKCU\...\Run.
bool IsStartupEnabled();
bool SetStartupEnabled(bool enable);
// If startup is enabled but points to an old exe location, fix it.
void RefreshStartupPath();

// "Ctrl+Alt+X" style text for a hotkey; empty string if none.
std::wstring HotkeyToString(UINT vk, UINT mods);
