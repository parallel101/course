template <class T>
struct atomic {
    std::mutex mtx;
    int cnt;

    int store(int val) {
        std::lock_guard _(mtx);
        cnt = val;
    }

    int load() const {
        std::lock_guard _(mtx);
        return cnt;
    }

    int fetch_add(int val) {
        std::lock_guard _(mtx);
        int old = cnt;
        cnt += val;
        return old;
    }

    int exchange(int val) {
        std::lock_guard _(mtx);
        int old = cnt;
        cnt = val;
        return old;
    }

    bool compare_exchange_strong(int &old, int val) {
        std::lock_guard _(mtx);
        if (cnt == old) {
            cnt = val;
            return true;
        } else {
            old = cnt;
            return false;
        }
    }
};
