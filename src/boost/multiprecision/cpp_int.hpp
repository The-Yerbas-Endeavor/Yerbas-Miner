#pragma once

// Minimal compatibility shim for the handful of boost::multiprecision::cpp_int
// operations used by Yerbas-Miner's Stratum target calculation. This keeps the
// miner self-contained on Windows while preserving the existing call sites.
// It is intentionally limited to unsigned 256-bit arithmetic.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace boost::multiprecision {

class cpp_int {
public:
    cpp_int() = default;
    cpp_int(std::uint64_t value)
    {
        for (std::size_t i = 0; i < 8; ++i) {
            bytes_[i] = static_cast<std::uint8_t>(value >> (i * 8));
        }
    }

    cpp_int& operator<<=(unsigned int bits)
    {
        if (bits == 0) return *this;
        if (bits >= 256) {
            bytes_.fill(0);
            return *this;
        }

        const unsigned int byte_shift = bits / 8;
        const unsigned int bit_shift = bits % 8;
        std::array<std::uint8_t, 32> out{};

        for (int i = 31; i >= 0; --i) {
            const int src = i - static_cast<int>(byte_shift);
            if (src < 0) continue;

            std::uint16_t value = static_cast<std::uint16_t>(bytes_[static_cast<std::size_t>(src)]) << bit_shift;
            out[static_cast<std::size_t>(i)] |= static_cast<std::uint8_t>(value & 0xffU);
            if (bit_shift != 0 && i + 1 < 32) {
                out[static_cast<std::size_t>(i + 1)] |= static_cast<std::uint8_t>(value >> 8);
            }
        }
        bytes_ = out;
        return *this;
    }

    cpp_int& operator>>=(unsigned int bits)
    {
        if (bits == 0) return *this;
        if (bits >= 256) {
            bytes_.fill(0);
            return *this;
        }

        const unsigned int byte_shift = bits / 8;
        const unsigned int bit_shift = bits % 8;
        std::array<std::uint8_t, 32> out{};

        for (std::size_t i = 0; i < 32; ++i) {
            const std::size_t src = i + byte_shift;
            if (src >= 32) continue;

            std::uint16_t value = bytes_[src];
            if (bit_shift != 0 && src + 1 < 32) {
                value |= static_cast<std::uint16_t>(bytes_[src + 1]) << 8;
            }
            out[i] = static_cast<std::uint8_t>((value >> bit_shift) & 0xffU);
        }
        bytes_ = out;
        return *this;
    }

    cpp_int& operator+=(std::uint64_t value)
    {
        std::uint64_t carry = value;
        for (std::size_t i = 0; i < 32 && carry != 0; ++i) {
            const std::uint64_t sum = static_cast<std::uint64_t>(bytes_[i]) + (carry & 0xffU);
            bytes_[i] = static_cast<std::uint8_t>(sum & 0xffU);
            carry = (carry >> 8) + (sum >> 8);
        }
        return *this;
    }

    friend cpp_int operator*(const cpp_int& lhs, std::uint64_t rhs)
    {
        cpp_int out;
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < 32; ++i) {
            // rhs is 1,000,000 in Yerbas-Miner. The product therefore remains
            // comfortably within uint64_t for each base-256 digit step.
            const std::uint64_t product = static_cast<std::uint64_t>(lhs.bytes_[i]) * rhs + carry;
            out.bytes_[i] = static_cast<std::uint8_t>(product & 0xffU);
            carry = product >> 8;
        }
        return out;
    }

    friend cpp_int operator/(const cpp_int& lhs, std::uint64_t divisor)
    {
        if (divisor == 0) throw std::runtime_error("cpp_int division by zero");

        // Base-256 long division. Pool share difficulties are far below the
        // limit where remainder*256 can overflow uint64_t. Guard explicitly
        // rather than silently producing a bad mining target.
        if (divisor > (std::numeric_limits<std::uint64_t>::max() >> 8)) {
            throw std::runtime_error("Stratum difficulty is too large for fixed-width target conversion");
        }

        cpp_int out;
        std::uint64_t remainder = 0;
        for (int i = 31; i >= 0; --i) {
            const std::uint64_t current = (remainder << 8) | lhs.bytes_[static_cast<std::size_t>(i)];
            out.bytes_[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(current / divisor);
            remainder = current % divisor;
        }
        return out;
    }

    friend std::uint64_t operator&(const cpp_int& lhs, std::uint64_t rhs)
    {
        std::uint64_t low = 0;
        for (std::size_t i = 0; i < 8; ++i) {
            low |= static_cast<std::uint64_t>(lhs.bytes_[i]) << (i * 8);
        }
        return low & rhs;
    }

private:
    std::array<std::uint8_t, 32> bytes_{};
};

} // namespace boost::multiprecision
