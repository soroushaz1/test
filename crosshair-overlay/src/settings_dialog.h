#pragma once

#include <windows.h>

// Opens the (modeless) settings window, or brings it to the front.
void ShowSettingsDialog();

// Handle of the open settings window, or nullptr. Used for IsDialogMessage.
HWND SettingsDialogHandle();

// Closes the window, discarding unsaved changes.
void CloseSettingsDialog();
