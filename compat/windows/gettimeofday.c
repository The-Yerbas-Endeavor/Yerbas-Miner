#ifdef _WIN32

#include <winsock2.h>
#include <windows.h>
#include <stdint.h>

struct timezone;

int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    (void)tz;

    if (tv == NULL) {
        return -1;
    }

    FILETIME ft;
    ULARGE_INTEGER uli;
    GetSystemTimeAsFileTime(&ft);
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;

    // Windows FILETIME is 100-ns ticks since 1601-01-01 UTC.
    // Unix time is seconds since 1970-01-01 UTC.
    const uint64_t epoch_offset_100ns = 116444736000000000ULL;
    const uint64_t unix_100ns = uli.QuadPart - epoch_offset_100ns;

    tv->tv_sec = (long)(unix_100ns / 10000000ULL);
    tv->tv_usec = (long)((unix_100ns % 10000000ULL) / 10ULL);
    return 0;
}

#endif
