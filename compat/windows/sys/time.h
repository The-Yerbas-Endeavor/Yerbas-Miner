#pragma once

// Minimal POSIX sys/time.h compatibility for the imported OpenAES source.
// OpenAES includes this header on every platform, but Yerbas' current build
// path does not require any POSIX-only functionality from it on Windows.

#ifdef _WIN32
#include <winsock2.h>
#else
#include_next <sys/time.h>
#endif
