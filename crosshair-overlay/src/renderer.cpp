#include "renderer.h"

#include <algorithm>
#include <cmath>

// The crosshair is described as a union of simple shapes (boxes, a disk, a
// ring). For every pixel we compute the signed distance to that union, which
// gives anti-aliased edges and a perfectly even outline (the outline is just
// the shape "grown" by the outline thickness). Axis-aligned edges sit exactly
// on pixel boundaries, so un-rotated crosshairs stay razor sharp.

namespace {

constexpr float kFar = 1.0e9f;
constexpr float kSqrt2 = 1.41421356f;

float Clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

// Signed distance from (px, py) to an axis-aligned box centered at the origin.
float SdBox(float px, float py, float hx, float hy) {
    const float qx = std::fabs(px) - hx;
    const float qy = std::fabs(py) - hy;
    const float ox = std::max(qx, 0.f);
    const float oy = std::max(qy, 0.f);
    return std::sqrt(ox * ox + oy * oy) + std::min(std::max(qx, qy), 0.f);
}

struct Shape {
    enum Kind { kBox, kDisk, kRing } kind;
    float cx, cy;          // center in canvas coordinates
    float a, b;            // box: half extents; disk: radius; ring: radius & half thickness
    float cs = 1.f, sn = 0.f;   // box rotation

    float Distance(float px, float py) const {
        const float dx = px - cx, dy = py - cy;
        switch (kind) {
            case kBox:
                return SdBox(dx * cs + dy * sn, -dx * sn + dy * cs, a, b);
            case kDisk:
                return std::sqrt(dx * dx + dy * dy) - a;
            case kRing:
                return std::fabs(std::sqrt(dx * dx + dy * dy) - a) - b;
        }
        return kFar;
    }

    // Half size of the axis-aligned bounding box.
    void Extent(float& ex, float& ey) const {
        ex = ey = 0.f;
        switch (kind) {
            case kBox:
                ex = std::fabs(cs) * a + std::fabs(sn) * b;
                ey = std::fabs(sn) * a + std::fabs(cs) * b;
                return;
            case kDisk:
                ex = ey = a;
                return;
            case kRing:
                ex = ey = a + b;
                return;
        }
    }
};

// Exact sin/cos for multiples of 90 degrees so un-rotated shapes stay crisp.
void RotationOf(int degrees, float& cs, float& sn) {
    switch (((degrees % 360) + 360) % 360) {
        case 0: cs = 1.f; sn = 0.f; return;
        case 90: cs = 0.f; sn = 1.f; return;
        case 180: cs = -1.f; sn = 0.f; return;
        case 270: cs = 0.f; sn = -1.f; return;
    }
    const double rad = degrees * 3.14159265358979323846 / 180.0;
    cs = static_cast<float>(std::cos(rad));
    sn = static_cast<float>(std::sin(rad));
}

// Even sizes are centered on a pixel corner, odd sizes on a pixel center, so
// that edges land on pixel boundaries.
float CenterFor(int center, int size) { return center + ((size & 1) ? 0.5f : 0.f); }

float OutlineWidth(const Settings& s) { return s.outline ? static_cast<float>(s.outlineThickness) : 0.f; }

}  // namespace

int CrosshairCanvasSize(const Settings& s) {
    float extent = 1.f;
    if (s.showCross) {
        const float h = s.thickness * 0.5f;
        const float reach = h + s.gap + s.length;
        const bool axisAligned = (s.rotation % 90) == 0;
        extent = std::max(extent, axisAligned ? std::max(reach, h) : std::hypot(reach, h));
    }
    if (s.showDot) extent = std::max(extent, s.dotSize * 0.5f * kSqrt2);
    if (s.showCircle) extent = std::max(extent, s.circleRadius + s.circleThickness * 0.5f);

    // +0.5 for the half-pixel center shift, +2 pixels of safety margin.
    const int r = static_cast<int>(std::ceil(extent + 0.5f + OutlineWidth(s) + 2.f));
    return 2 * r + 1;
}

void RenderCrosshair(const Settings& s, CrosshairImage& out) {
    const int size = CrosshairCanvasSize(s);
    const int c = size / 2;
    out.size = size;
    out.center = c;
    out.pixels.assign(static_cast<size_t>(size) * size, 0u);

    std::vector<Shape> shapes;
    const float ccx = CenterFor(c, s.thickness);
    const float ccy = ccx;

    if (s.showCross) {
        float cs, sn;
        RotationOf(s.rotation, cs, sn);
        // (ax, ay): arm center relative to the cross center, before rotation.
        auto addBox = [&](float ax, float ay, float hx, float hy) {
            shapes.push_back({Shape::kBox, ccx + ax * cs - ay * sn, ccy + ax * sn + ay * cs, hx, hy, cs, sn});
        };
        const float h = s.thickness * 0.5f;
        if (s.length > 0) {
            const float half = s.length * 0.5f;
            const float mid = h + s.gap + half;
            addBox(mid, 0.f, half, h);              // right
            addBox(-mid, 0.f, half, h);             // left
            addBox(0.f, mid, h, half);              // bottom (screen y grows downward)
            if (!s.tStyle) addBox(0.f, -mid, h, half);  // top
        }
        if (s.gap == 0) addBox(0.f, 0.f, h, h);     // join the arms into a solid "+"
    }

    if (s.showCircle) {
        shapes.push_back({Shape::kRing, ccx, ccy, static_cast<float>(s.circleRadius), s.circleThickness * 0.5f});
    }

    if (s.showDot) {
        const float dc = CenterFor(c, s.dotSize);
        const float r = s.dotSize * 0.5f;
        shapes.push_back({s.dotRound ? Shape::kDisk : Shape::kBox, dc, dc, r, r});
    }

    if (shapes.empty()) return;

    // Distance field of the union of all shapes; each shape only touches the
    // pixels inside its own bounding box (plus outline margin).
    const float ow = OutlineWidth(s);
    const float margin = ow + 1.5f;
    std::vector<float> dist(static_cast<size_t>(size) * size, kFar);
    for (const Shape& sh : shapes) {
        float ex, ey;
        sh.Extent(ex, ey);
        const int x0 = std::max(0, static_cast<int>(std::floor(sh.cx - ex - margin)));
        const int x1 = std::min(size - 1, static_cast<int>(std::ceil(sh.cx + ex + margin)));
        const int y0 = std::max(0, static_cast<int>(std::floor(sh.cy - ey - margin)));
        const int y1 = std::min(size - 1, static_cast<int>(std::ceil(sh.cy + ey + margin)));
        for (int y = y0; y <= y1; ++y) {
            float* row = &dist[static_cast<size_t>(y) * size];
            const float py = y + 0.5f;
            for (int x = x0; x <= x1; ++x) {
                const float d = sh.Distance(x + 0.5f, py);
                if (d < row[x]) row[x] = d;
            }
        }
    }

    const float fr = GetRValue(s.color), fg = GetGValue(s.color), fb = GetBValue(s.color);
    const float orr = GetRValue(s.outlineColor), og = GetGValue(s.outlineColor), ob = GetBValue(s.outlineColor);

    for (size_t i = 0; i < dist.size(); ++i) {
        const float d = dist[i];
        if (d > margin) continue;   // transparent
        // Coverage: the distance is measured from the pixel center, so a
        // pixel is fully covered at d <= -0.5 and empty at d >= +0.5.
        const float fill = Clamp01(0.5f - d);
        const float line = (ow > 0.f) ? Clamp01(0.5f - (d - ow)) * (1.f - fill) : 0.f;
        const float a = fill + line;
        if (a <= 0.f) continue;
        // Premultiplied BGRA: fill drawn over the outline.
        const uint32_t A = static_cast<uint32_t>(a * 255.f + 0.5f);
        const uint32_t R = static_cast<uint32_t>(fr * fill + orr * line + 0.5f);
        const uint32_t G = static_cast<uint32_t>(fg * fill + og * line + 0.5f);
        const uint32_t B = static_cast<uint32_t>(fb * fill + ob * line + 0.5f);
        out.pixels[i] = (A << 24) | (std::min(R, A) << 16) | (std::min(G, A) << 8) | std::min(B, A);
    }
}
