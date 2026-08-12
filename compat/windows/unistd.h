#pragma once

#ifdef _WIN32
#include <process.h>
#else
#include_next <unistd.h>
#endif
