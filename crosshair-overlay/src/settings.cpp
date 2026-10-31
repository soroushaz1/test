#include "settings.h"

#include <shlobj.h>

#include <algorithm>
#include <cwchar>

namespace {

const wchar_t kAppDirName[] = L"CrosshairOverlay";
const wchar_t kRunKey[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
const wchar_t kRunValue[] = L"CrosshairOverlay";
const wchar_t kSecCrosshair[] = L"Crosshair";
const wchar_t kSecBehavior[] = L"Behavior";

std::wstring ExePath() {
    wchar_t buf[MAX_PATH * 2];
    DWORD n = GetModuleFileNameW(nullptr, buf, ARRAYSIZE(buf));
    return std::wstring(buf, n);
}

std::wstring DirOf(const std::wstring& path) {
    size_t pos = path.find_last_of(L"\\/");
    return pos == std::wstring::npos ? std::wstring() : path.substr(0, pos);
}

bool FileExists(const std::wstring& path) {
    DWORD a = GetFileAttributesW(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

int ReadInt(const std::wstring& file, const wchar_t* sec, const wchar_t* key, int def) {
    wchar_t buf[64];
    GetPrivateProfileStringW(sec, key, L"", buf, ARRAYSIZE(buf), file.c_str());
    if (!buf[0]) return def;
    wchar_t* end = nullptr;
    long v = wcstol(buf, &end, 0);   // base 0: accepts decimal and 0x.. hex
    return (end == buf) ? def : static_cast<int>(v);
}

bool ReadBool(const std::wstring& file, const wchar_t* sec, const wchar_t* key, bool def) {
    return ReadInt(file, sec, key, def ? 1 : 0) != 0;
}

COLORREF ReadColor(const std::wstring& file, const wchar_t* sec, const wchar_t* key, COLORREF def) {
    wchar_t buf[32];
    GetPrivateProfileStringW(sec, key, L"", buf, ARRAYSIZE(buf), file.c_str());
    const wchar_t* p = buf;
    if (*p == L'#') ++p;
    if (wcslen(p) != 6) return def;
    wchar_t* end = nullptr;
    unsigned long v = wcstoul(p, &end, 16);
    if (*end) return def;
    return RGB((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
}

void WriteInt(const std::wstring& file, const wchar_t* sec, const wchar_t* key, int v) {
    wchar_t buf[32];
    swprintf(buf, ARRAYSIZE(buf), L"%d", v);
    WritePrivateProfileStringW(sec, key, buf, file.c_str());
}

void WriteColor(const std::wstring& file, const wchar_t* sec, const wchar_t* key, COLORREF c) {
    wchar_t buf[16];
    swprintf(buf, ARRAYSIZE(buf), L"#%02X%02X%02X", GetRValue(c), GetGValue(c), GetBValue(c));
    WritePrivateProfileStringW(sec, key, buf, file.c_str());
}

std::wstring QuotedExePath() { return L"\"" + ExePath() + L"\""; }

}  // namespace

void Settings::Clamp() {
    using namespace limits;
    length = std::clamp(length, kLengthMin, kLengthMax);
    thickness = std::clamp(thickness, kThicknessMin, kThicknessMax);
    gap = std::clamp(gap, kGapMin, kGapMax);
    rotation = std::clamp(rotation, kRotationMin, kRotationMax);
    dotSize = std::clamp(dotSize, kDotMin, kDotMax);
    circleRadius = std::clamp(circleRadius, kRadiusMin, kRadiusMax);
    circleThickness = std::clamp(circleThickness, kCircleThickMin, kCircleThickMax);
    opacity = std::clamp(opacity, kOpacityMin, kOpacityMax);
    outlineThickness = std::clamp(outlineThickness, kOutlineMin, kOutlineMax);
    offsetX = std::clamp(offsetX, kOffsetMin, kOffsetMax);
    offsetY = std::clamp(offsetY, kOffsetMin, kOffsetMax);
    hotkeyMods &= (MOD_ALT | MOD_CONTROL | MOD_SHIFT | MOD_WIN);
    if (hotkeyVk > 0xFE) hotkeyVk = 0;
}

std::wstring SettingsFilePath() {
    std::wstring portable = DirOf(ExePath()) + L"\\settings.ini";
    if (FileExists(portable)) return portable;

    wchar_t appData[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, SHGFP_TYPE_CURRENT, appData))) {
        std::wstring dir = std::wstring(appData) + L"\\" + kAppDirName;
        CreateDirectoryW(dir.c_str(), nullptr);
        return dir + L"\\settings.ini";
    }
    return portable;
}

bool LoadSettings(Settings& s) {
    const std::wstring f = SettingsFilePath();
    const bool existed = FileExists(f);
    const Settings d;  // defaults
    const wchar_t* C = kSecCrosshair;
    const wchar_t* B = kSecBehavior;

    s.showCross = ReadBool(f, C, L"ShowCross", d.showCross);
    s.length = ReadInt(f, C, L"Length", d.length);
    s.thickness = ReadInt(f, C, L"Thickness", d.thickness);
    s.gap = ReadInt(f, C, L"Gap", d.gap);
    s.rotation = ReadInt(f, C, L"Rotation", d.rotation);
    s.tStyle = ReadBool(f, C, L"TStyle", d.tStyle);
    s.showDot = ReadBool(f, C, L"ShowDot", d.showDot);
    s.dotSize = ReadInt(f, C, L"DotSize", d.dotSize);
    s.dotRound = ReadBool(f, C, L"DotRound", d.dotRound);
    s.showCircle = ReadBool(f, C, L"ShowCircle", d.showCircle);
    s.circleRadius = ReadInt(f, C, L"CircleRadius", d.circleRadius);
    s.circleThickness = ReadInt(f, C, L"CircleThickness", d.circleThickness);
    s.color = ReadColor(f, C, L"Color", d.color);
    s.opacity = ReadInt(f, C, L"Opacity", d.opacity);
    s.outline = ReadBool(f, C, L"Outline", d.outline);
    s.outlineColor = ReadColor(f, C, L"OutlineColor", d.outlineColor);
    s.outlineThickness = ReadInt(f, C, L"OutlineThickness", d.outlineThickness);
    s.offsetX = ReadInt(f, C, L"OffsetX", d.offsetX);
    s.offsetY = ReadInt(f, C, L"OffsetY", d.offsetY);

    s.hideWhenCursorHidden = ReadBool(f, B, L"HideWhenCursorHidden", d.hideWhenCursorHidden);
    s.hotkeyVk = static_cast<UINT>(ReadInt(f, B, L"HotkeyVk", static_cast<int>(d.hotkeyVk)));
    s.hotkeyMods = static_cast<UINT>(ReadInt(f, B, L"HotkeyMods", static_cast<int>(d.hotkeyMods)));

    s.Clamp();
    return existed;
}

bool SaveSettings(const Settings& s) {
    const std::wstring f = SettingsFilePath();
    const wchar_t* C = kSecCrosshair;
    const wchar_t* B = kSecBehavior;

    WriteInt(f, C, L"ShowCross", s.showCross);
    WriteInt(f, C, L"Length", s.length);
    WriteInt(f, C, L"Thickness", s.thickness);
    WriteInt(f, C, L"Gap", s.gap);
    WriteInt(f, C, L"Rotation", s.rotation);
    WriteInt(f, C, L"TStyle", s.tStyle);
    WriteInt(f, C, L"ShowDot", s.showDot);
    WriteInt(f, C, L"DotSize", s.dotSize);
    WriteInt(f, C, L"DotRound", s.dotRound);
    WriteInt(f, C, L"ShowCircle", s.showCircle);
    WriteInt(f, C, L"CircleRadius", s.circleRadius);
    WriteInt(f, C, L"CircleThickness", s.circleThickness);
    WriteColor(f, C, L"Color", s.color);
    WriteInt(f, C, L"Opacity", s.opacity);
    WriteInt(f, C, L"Outline", s.outline);
    WriteColor(f, C, L"OutlineColor", s.outlineColor);
    WriteInt(f, C, L"OutlineThickness", s.outlineThickness);
    WriteInt(f, C, L"OffsetX", s.offsetX);
    WriteInt(f, C, L"OffsetY", s.offsetY);

    WriteInt(f, B, L"HideWhenCursorHidden", s.hideWhenCursorHidden);
    WriteInt(f, B, L"HotkeyVk", static_cast<int>(s.hotkeyVk));
    WriteInt(f, B, L"HotkeyMods", static_cast<int>(s.hotkeyMods));

    // Flush the profile cache so the file is written right away, and verify
    // that it actually landed on disk.
    WritePrivateProfileStringW(nullptr, nullptr, nullptr, f.c_str());
    return FileExists(f);
}

bool IsStartupEnabled() {
    DWORD size = 0;
    return RegGetValueW(HKEY_CURRENT_USER, kRunKey, kRunValue, RRF_RT_REG_SZ, nullptr, nullptr,
                        &size) == ERROR_SUCCESS;
}

bool SetStartupEnabled(bool enable) {
    if (!enable) {
        LSTATUS st = RegDeleteKeyValueW(HKEY_CURRENT_USER, kRunKey, kRunValue);
        return st == ERROR_SUCCESS || st == ERROR_FILE_NOT_FOUND;
    }
    std::wstring cmd = QuotedExePath();
    return RegSetKeyValueW(HKEY_CURRENT_USER, kRunKey, kRunValue, REG_SZ, cmd.c_str(),
                           static_cast<DWORD>((cmd.size() + 1) * sizeof(wchar_t))) == ERROR_SUCCESS;
}

void RefreshStartupPath() {
    wchar_t buf[MAX_PATH * 2 + 4];
    DWORD size = sizeof(buf);
    if (RegGetValueW(HKEY_CURRENT_USER, kRunKey, kRunValue, RRF_RT_REG_SZ, nullptr, buf, &size) !=
        ERROR_SUCCESS)
        return;
    if (_wcsicmp(buf, QuotedExePath().c_str()) != 0) SetStartupEnabled(true);
}

std::wstring HotkeyToString(UINT vk, UINT mods) {
    if (!vk) return std::wstring();
    std::wstring s;
    if (mods & MOD_WIN) s += L"Win+";
    if (mods & MOD_CONTROL) s += L"Ctrl+";
    if (mods & MOD_ALT) s += L"Alt+";
    if (mods & MOD_SHIFT) s += L"Shift+";

    UINT scan = MapVirtualKeyW(vk, MAPVK_VK_TO_VSC);
    switch (vk) {  // keys that need the "extended" bit to get the right name
        case VK_LEFT: case VK_RIGHT: case VK_UP: case VK_DOWN:
        case VK_PRIOR: case VK_NEXT: case VK_HOME: case VK_END:
        case VK_INSERT: case VK_DELETE: case VK_DIVIDE: case VK_NUMLOCK:
            scan |= 0x100;
            break;
    }
    wchar_t name[64] = {};
    if (scan && GetKeyNameTextW(static_cast<LONG>(scan << 16), name, ARRAYSIZE(name)) > 0) {
        s += name;
    } else {
        wchar_t hex[16];
        swprintf(hex, ARRAYSIZE(hex), L"VK 0x%02X", vk);
        s += hex;
    }
    return s;
}
