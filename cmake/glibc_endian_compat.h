#pragma once

// Modern glibc exposes endian conversion helpers such as htole16/htobe32 as
// macros. Yerbas Core's legacy compat/endian.h provides inline functions with
// the same names. Include the system header once, then undefine the macros so
// Core can declare its compatibility functions without macro expansion into
// glibc internals such as __uint16_identity/__bswap_32.
#if defined(__linux__)
#include <endian.h>

#ifdef htobe16
#undef htobe16
#endif
#ifdef htole16
#undef htole16
#endif
#ifdef be16toh
#undef be16toh
#endif
#ifdef le16toh
#undef le16toh
#endif
#ifdef htobe32
#undef htobe32
#endif
#ifdef htole32
#undef htole32
#endif
#ifdef be32toh
#undef be32toh
#endif
#ifdef le32toh
#undef le32toh
#endif
#ifdef htobe64
#undef htobe64
#endif
#ifdef htole64
#undef htole64
#endif
#ifdef be64toh
#undef be64toh
#endif
#ifdef le64toh
#undef le64toh
#endif
#endif

// Yerbas Core's legacy CryptoNight implementation intentionally type-puns
// 16-byte state/scratchpad blocks through uint8_t*, uint32_t* and uint64_t*.
// With modern GCC at -O3 those accesses violate strict-aliasing assumptions and
// can produce input-dependent hashes that differ from the pool-proven CUDA
// implementation. The original algorithm relies on the byte/word views
// aliasing exactly, so compile the pinned Core reference translation units with
// strict-aliasing optimizations disabled. This header is force-included only on
// the yerbas-ghostrider-reference target.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize ("no-strict-aliasing")
#endif
