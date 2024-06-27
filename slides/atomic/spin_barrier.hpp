#pragma once

#include <atomic>
#include <cstdint>

struct SpinBarrier {
    explicit SpinBarrier(std::size_t n) noexcept
        : m_top_waiting((std::uint32_t)n - 1),
          m_num_waiting(0),
          m_sync_flip(0) {}

    bool arrive_and_wait() noexcept {
        bool old_flip = m_sync_flip.load(std::memory_order_relaxed);
        if (m_num_waiting.fetch_add(1, std::memory_order_relaxed) == m_top_waiting) {
            m_num_waiting.store(0, std::memory_order_relaxed);
            m_sync_flip.store(!old_flip, std::memory_order_release);
#if __cpp_lib_atomic_wait
            m_sync_flip.notify_all();
#endif
            return true;
        } else {
#if __cpp_lib_atomic_wait
            std::uint32_t retries = 255;
            do {
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
            } while (--retries);
#else
            while (m_sync_flip.load(std::memory_order_acquire) == old_flip)
                ;
#endif
            m_sync_flip.wait(old_flip, std::memory_order_acquire);
            return false;
        }
    }

    bool arrive_and_drop() noexcept {
        bool old_flip = m_sync_flip.load(std::memory_order_relaxed);
        if (m_num_waiting.fetch_add(1, std::memory_order_relaxed) == m_top_waiting) {
            m_num_waiting.store(0, std::memory_order_relaxed);
            m_sync_flip.store(!old_flip, std::memory_order_release);
#if __cpp_lib_atomic_wait
            m_sync_flip.notify_all();
#endif
            return true;
        } else {
            return false;
        }
    }

private:
    std::uint32_t const m_top_waiting;
    std::atomic<std::uint32_t> m_num_waiting;
    std::atomic<bool> m_sync_flip;
};

/* #if __cplusplus >= 202002L && __has_include(<barrier>) */
/* #include <barrier> */
/* #endif */
/* #if !__cpp_lib_barrier */
/* using StdBarrier = SpinBarrier; */
/* #else */
/* using StdBarrier = std::barrier<>; */
/* #endif */
