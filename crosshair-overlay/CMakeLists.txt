cmake_minimum_required(VERSION 3.16)
project(CrosshairOverlay LANGUAGES CXX RC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(CrosshairOverlay WIN32
    src/main.cpp
    src/overlay.cpp
    src/renderer.cpp
    src/settings.cpp
    src/settings_dialog.cpp
    src/app.rc
)

target_compile_definitions(CrosshairOverlay PRIVATE
    UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 WINVER=0x0A00)

target_link_libraries(CrosshairOverlay PRIVATE
    comctl32 comdlg32 shell32 gdi32 user32 advapi32 msimg32 winmm)

if(MSVC)
    # app.rc already embeds app.manifest; don't let the linker add a second one.
    target_link_options(CrosshairOverlay PRIVATE /MANIFEST:NO)
    target_compile_options(CrosshairOverlay PRIVATE /W4 /utf-8)
    set_property(TARGET CrosshairOverlay PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_compile_options(CrosshairOverlay PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    target_link_options(CrosshairOverlay PRIVATE -municode -static -static-libgcc -static-libstdc++)
endif()
