#pragma once

// Minimal compatibility shim for the handful of boost::multiprecision::cpp_int
// operations used by Yerbas-Miner's Stratum target calculation. This keeps the
// miner self-contained on Windows while preserving the existing call sites.
//
// The final mining target is 256 bits, but GhostRider's fixed-point difficulty
// conversion can temporarily exceed 256 bits before division. Keep a small
// widened working area so those intermediate products do not wrap.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace boost::multiprecision {

class cpp_int {
public:
    static constexpr std::size_t kBytes = 48; // 384-bit working precision
    static constexpr unsigned int kBits = static_cast<unsigned int>(kBytes * 8);

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
        if (bits >= kBits) {
            bytes_.fill(0);
            return *this;
        }

        const unsigned int byte_shift = bits / 8;
        const unsigned int bit_shift = bits % 8;
        std::array<std::uint8_t, kBytes> out{};

        for (int i = static_cast<int>(kBytes) - 1; i >= 0; --i) {
            const int src = i - static_cast<int>(byte_shift);
            if (src < 0) continue;

            const std::uint16_t value =
                static_cast<std::uint16_t>(bytes_[static_cast<std::size_t>(src)]) << bit_shift;
            out[static_cast<std::size_t>(i)] |= static_cast<std::uint8_t>(value & 0xffU);
            if (bit_shift != 0 && i + 1 < static_cast<int>(kBytes)) {
                out[static_cast<std::size_t>(i + 1)] |= static_cast<std::uint8_t>(value >> 8);
            }
        }
        bytes_ = out;
        return *this;
    }

    cpp_int& operator>>=(unsigned int bits)
    {
        if (bits == 0) return *this;
        if (bits >= kBits) {
            bytes_.fill(0);
            return *this;
        }

        const unsigned int byte_shift = bits / 8;
        const unsigned int bit_shift = bits % 8;
        std::array<std::uint8_t, kBytes> out{};

        for (std::size_t i = 0; i < kBytes; ++i) {
            const std::size_t src = i + byte_shift;
            if (src >= kBytes) continue;

            std::uint16_t value = bytes_[src];
            if (bit_shift != 0 && src + 1 < kBytes) {
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
        for (std::size_t i = 0; i < kBytes && carry != 0; ++i) {
            const std::uint64_t sum = static_cast<std::uint64_t>(bytes_[i]) + (carry & 0xffU);
            bytes_[i] = static_cast<std::uint8_t>(sum & 0xffU);
            carry = (carry >> 8) + (sum >> 8);
        }
        return *this;
    }

    friend cpp_int operator<<(cpp_int lhs, unsigned int bits)
    {
        lhs <<= bits;
        return lhs;
    }

    friend cpp_int operator-(cpp_int lhs, std::uint64_t rhs)
    {
        std::uint64_t borrow = rhs;
        for (std::size_t i = 0; i < kBytes && borrow != 0; ++i) {
            const std::uint64_t sub = borrow & 0xffU;
            const std::uint64_t current = lhs.bytes_[i];
            if (current >= sub) {
                lhs.bytes_[i] = static_cast<std::uint8_t>(current - sub);
                borrow >>= 8;
            } else {
                lhs.bytes_[i] = static_cast<std::uint8_t>(256U + current - sub);
                borrow = (borrow >> 8) + 1U;
            }
        }
        return lhs;
    }

    friend bool operator>(const cpp_int& lhs, const cpp_int& rhs)
    {
        for (int i = static_cast<int>(kBytes) - 1; i >= 0; --i) {
            const auto index = static_cast<std::size_t>(i);
            if (lhs.bytes_[index] > rhs.bytes_[index]) return true;
            if (lhs.bytes_[index] < rhs.bytes_[index]) return false;
        }
        return false;
    }

    friend cpp_int operator*(const cpp_int& lhs, std::uint64_t rhs)
    {
        cpp_int out;
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < kBytes; ++i) {
            const std::uint64_t product = static_cast<std::uint64_t>(lhs.bytes_[i]) * rhs + carry;
            out.bytes_[i] = static_cast<std::uint8_t>(product & 0xffU);
            carry = product >> 8;
        }
        if (carry != 0) {
            throw std::runtime_error("cpp_int working precision exceeded during target multiplication");
        }
        return out;
    }

    friend cpp_int operator/(const cpp_int& lhs, std::uint64_t divisor)
    {
        if (divisor == 0) throw std::runtime_error("cpp_int division by zero");

        if (divisor > (std::numeric_limits<std::uint64_t>::max() >> 8)) {
            throw std::runtime_error("Stratum difficulty is too large for fixed-width target conversion");
        }

        cpp_int out;
        std::uint64_t remainder = 0;
        for (int i = static_cast<int>(kBytes) - 1; i >= 0; --i) {
            const std::uint64_t current =
                (remainder << 8) | lhs.bytes_[static_cast<std::size_t>(i)];
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
    std::array<std::uint8_t, kBytes> bytes_{};
};

} // namespace boost::multiprecision
