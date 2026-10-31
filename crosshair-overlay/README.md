# Crosshair Overlay

یک برنامه‌ی کوچک ویندوزی که یک **کراس‌هیر (نشانگر) قابل تنظیم** را دقیقاً در **محل ماوس** و **بالای همه‌ی پنجره‌ها** نشان می‌دهد.

- یک فایل `exe` تکی، بدون نصب و بدون نیاز به هیچ کتابخانه‌ی جانبی (فقط Win32 API خود ویندوز)
- کلیک‌ها از روی کراس‌هیر رد می‌شوند (click-through) و هیچ‌وقت فوکوس پنجره‌ها را نمی‌گیرد
- کراس‌هیر با حرکت ماوس جابه‌جا می‌شود (موقعیت ماوس تا ۱۰۰۰ بار در ثانیه خوانده می‌شود)
- در حالت بیکار مصرف CPU تقریباً صفر است
- سازگار با چند مانیتور و صفحه‌های High-DPI (اندازه‌ها بر حسب پیکسل واقعی صفحه)

## دانلود و اجرا

فایل آماده: [`bin/CrosshairOverlay.exe`](bin/CrosshairOverlay.exe) (ویندوز ۱۰ و ۱۱، ۶۴ بیتی)

۱. فایل را دانلود و اجرا کنید (نصب لازم نیست).
۲. چون فایل امضای دیجیتال ندارد، ممکن است ویندوز پیام SmartScreen نشان دهد. روی **More info** و بعد **Run anyway** بزنید.
۳. در اولین اجرا، پنجره‌ی تنظیمات باز می‌شود و آیکون برنامه در System Tray (کنار ساعت) قرار می‌گیرد.
   در ویندوز ۱۱ ممکن است آیکون داخل فلش `^` پنهان شده باشد.

## نحوه‌ی استفاده

| کار | روش |
| --- | --- |
| نمایش/مخفی کردن کراس‌هیر | کلید میان‌بر <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>X</kbd> (قابل تغییر) یا منوی آیکون Tray |
| باز کردن تنظیمات | کلیک چپ روی آیکون Tray، یا اجرای دوباره‌ی برنامه |
| منو (اجرا با ویندوز، خروج و …) | کلیک راست روی آیکون Tray |

همه‌ی تغییرات در پنجره‌ی تنظیمات **همان لحظه** روی کراس‌هیر اعمال می‌شوند. دکمه‌ی **OK** آن‌ها را ذخیره می‌کند و **Cancel** همه‌چیز را به حالت قبل برمی‌گرداند.

### تنظیمات قابل تغییر

- **خطوط (Cross lines):** طول، ضخامت، فاصله از مرکز (Gap)، چرخش (مثلاً ۴۵ درجه برای حالت X)، حالت T (بدون بازوی بالا)
- **نقطه‌ی مرکز (Center dot):** اندازه، گرد یا مربعی
- **دایره (Circle):** شعاع و ضخامت
- **رنگ‌ها:** رنگ اصلی، میزان شفافیت (Opacity)، حاشیه (Outline) با رنگ و ضخامت دلخواه
- **جابه‌جایی (Offset):** جابه‌جا کردن کراس‌هیر به چپ/راست/بالا/پایین (مثلاً برای تراز کردن دقیق با وسط بازی)
- **محل کراس‌هیر (Crosshair position):** جدول زیر را ببینید
- **رفتار:** کلید میان‌بر، اجرا همراه با ویندوز
- **پیش‌تنظیم‌ها (Preset):** چند طرح آماده برای شروع سریع

### محل کراس‌هیر (Crosshair position)

| گزینه | رفتار |
| --- | --- |
| **Auto: mouse, or center in games** (پیش‌فرض) | روی دسکتاپ دنبال ماوس می‌رود. داخل بازی، یعنی وقتی نشانگر ماوس مخفی است و بازی تمام‌صفحه است یا ماوس را داخل پنجره‌ی خودش قفل کرده، وسط پنجره‌ی بازی می‌ایستد |
| Always follow the mouse | همیشه روی محل ماوس (حتی وقتی ماوس مخفی است) |
| Mouse, hide when cursor is hidden | روی محل ماوس، و وقتی ماوس مخفی شود کراس‌هیر هم مخفی می‌شود |
| Center of the active window | همیشه وسط پنجره‌ای که فعال است |
| Center of the screen | همیشه وسط مانیتوری که پنجره‌ی فعال روی آن است |

در حالت Auto، مخفی شدن معمولی نشانگر (مثلاً گزینه‌ی «پنهان کردن نشانگر هنگام تایپ» ویندوز) باعث پریدن کراس‌هیر به وسط پنجره نمی‌شود.

## استفاده در بازی‌ها (مثل CS 1.6 و Warzone)

کراس‌هیر یک پنجره‌ی معمولی «همیشه رو» است. ویندوز فقط وقتی اجازه می‌دهد پنجره‌ای روی بازی دیده شود که بازی در حالت **Borderless** (تمام‌صفحه‌ی بدون حاشیه) یا **Windowed** باشد. در حالت **Exclusive Fullscreen** بازی کل صفحه را مستقیم در اختیار می‌گیرد و هیچ پنجره‌ی دیگری (از جمله این برنامه) روی آن نمایش داده نمی‌شود.

- **Warzone / Call of Duty:**
  Settings → Graphics → Display → **Display Mode = Fullscreen Borderless**
- **Counter-Strike 1.6:**
  ۱. در Steam روی بازی کلیک راست کنید، **Properties** را بزنید و در قسمت **Launch Options** این را بنویسید:
  ```
  -windowed -noborder
  ```
  ۲. داخل بازی از Options → Video، رزولوشن را برابر رزولوشن دسکتاپ بگذارید (مثلاً 1920×1080) تا بازی کل صفحه را بگیرد.

اگر کراس‌هیر دقیقاً روی وسط بازی نیفتاد، با **Offset** یکی دو پیکسل جابه‌جایش کنید.

این برنامه هیچ کاری با خود بازی ندارد. نه به حافظه‌ی بازی دسترسی دارد، نه چیزی به آن تزریق می‌کند و نه ورودی‌ها را دست‌کاری می‌کند. فقط یک پنجره‌ی شفاف بالای صفحه است. با این حال، استفاده از هر برنامه‌ی جانبی در بازی‌های آنلاینی که anti-cheat دارند به مسئولیت خودتان است.

تنظیمات در `%APPDATA%\CrosshairOverlay\settings.ini` ذخیره می‌شوند.
**حالت پرتابل:** اگر فایلی به اسم `settings.ini` کنار خود `exe` باشد، برنامه از همان فایل استفاده می‌کند.

### محدودیت‌ها

- بازی‌هایی که در حالت **Exclusive Fullscreen** اجرا می‌شوند اجازه‌ی نمایش پنجره‌ی دیگری روی خودشان نمی‌دهند. در این بازی‌ها حالت نمایش را روی **Borderless / Windowed Fullscreen** بگذارید (بخش «استفاده در بازی‌ها» را ببینید).
- چند بخش خاص ویندوز مثل منوی Start و Task Manager در لایه‌ای بالاتر از همه‌ی برنامه‌های عادی قرار دارند و کراس‌هیر زیر آن‌ها می‌رود.

## ساختن (Build) از روی سورس

### با Visual Studio یا CMake (روی ویندوز)

```bat
cmake -S . -B build
cmake --build build --config Release
```

یا در Visual Studio 2019/2022: **File → Open → Folder** و انتخاب همین پوشه (CMake خودکار شناسایی می‌شود).

### با MinGW-w64 (روی لینوکس یا MSYS2)

```bash
sudo apt install g++-mingw-w64-x86-64   # Ubuntu/Debian
./build.sh                              # خروجی: build/CrosshairOverlay.exe
```

## ساختار پروژه

```
src/main.cpp             نقطه‌ی شروع، آیکون Tray، کلید میان‌بر، دنبال کردن ماوس
src/overlay.*            پنجره‌ی شفاف، همیشه رو و click-through که کراس‌هیر را نشان می‌دهد
src/renderer.*           رسم کراس‌هیر (anti-aliased، با لبه‌های کاملاً تیز در حالت بدون چرخش)
src/settings.*           تنظیمات، ذخیره در فایل INI، اجرا همراه با ویندوز
src/settings_dialog.*    پنجره‌ی تنظیمات
src/app.rc               آیکون‌ها، طرح پنجره‌ی تنظیمات، manifest
tools/make_icons.py      ساخت فایل‌های آیکون (نیاز به Pillow)
```

---

## English summary

**Crosshair Overlay** draws a customizable crosshair at the mouse position, above all other windows. It's a single dependency-free Win32 executable.

- **Click-through overlay:** it never steals focus.
- **Smooth tracking:** the cursor is polled at 1 kHz while moving. It drops to about 67 Hz when idle.
- **Per-monitor DPI aware.**
- **Customizable:** lines (length, thickness, gap, rotation, T-style), center dot, circle, color, opacity, outline, offset and presets.
- **Global toggle hotkey:** default <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>X</kbd>.
- **Position modes:**
  - **Auto (default):** follows the mouse. When a game hides the cursor (fullscreen, or with the cursor confined to its window), the crosshair moves to the center of the game window.
  - **Always follow the mouse.**
  - **Mouse, hidden while the cursor is hidden.**
  - **Center of the active window.**
  - **Center of the screen.**
- **Stays on top of games:** the overlay is re-raised as soon as another window (e.g. a game) comes to the front.
- **Start with Windows.**

**Games:** run the game in **Borderless** or **Windowed** mode. Nothing external can draw over *exclusive* fullscreen.
- Warzone: *Display Mode = Fullscreen Borderless*.
- CS 1.6: Steam launch options `-windowed -noborder`, with the game resolution set to the desktop resolution.

The app never touches the game process; it is only a transparent, always-on-top window.

To use it, run `bin/CrosshairOverlay.exe`. Left-click the tray icon for settings and right-click it for the menu. Settings are stored in `%APPDATA%\CrosshairOverlay\settings.ini`, or in `settings.ini` next to the exe (portable mode).

To build it, use `cmake -S . -B build && cmake --build build --config Release` (MSVC), or `./build.sh` with MinGW-w64.
