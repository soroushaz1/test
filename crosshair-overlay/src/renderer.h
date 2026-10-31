#pragma once

#include <cstdint>
#include <vector>

#include "settings.h"

// Square canvas of premultiplied BGRA pixels (the format UpdateLayeredWindow
// and AlphaBlend expect). Pixel (center, center) is the mouse hotspot.
struct CrosshairImage {
    int size = 0;
    int center = 0;
    std::vector<uint32_t> pixels;   // size * size, top-down rows
};

// Canvas size needed for the given settings (always odd: 2 * center + 1).
int CrosshairCanvasSize(const Settings& s);

// Rasterizes the crosshair. Opacity is NOT baked in; it is applied as the
// layered window's constant alpha so changing it is cheap.
void RenderCrosshair(const Settings& s, CrosshairImage& out);
