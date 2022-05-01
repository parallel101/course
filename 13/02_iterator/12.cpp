#include <cstddef>

template <class T>
struct List {
    struct Node {
        T value;
        Node *next;
    };

    struct Iterator {
        Node *curr;

        Iterator &operator++() {
            curr = curr->next;
            return *this;
        }

        T &operator*() const {
            return curr->value;
        }

        bool operator!=(Iterator const &that) const {
            return curr != that.curr;
        }
    };

    Node *head;

    Iterator begin() { return {head}; }
    Iterator end() { return {nullptr}; }
};

template <class T>
struct Vector {
    struct Node {
        T value;
        Node *next;
    };

    struct Iterator {
        Node *curr;

        Iterator &operator++() {
            curr = curr->next;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            this->operator++();
            return tmp;
        }

        T &operator*() const {
            return curr->value;
        }

        bool operator!=(Iterator const &that) const {
            return curr != that.curr;
        }
    };

    Node *head;
    size_t size;

    Iterator begin() { return {head}; }
    Iterator end() { return {head + size}; }
};

void iterate_over_list(List<int> const &list) {
    for (auto curr = list.head; curr != nullptr; curr = curr->next) {
    }
}
