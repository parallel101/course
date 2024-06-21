#pragma once

#include <cstddef>
#include <cstdint>

enum class AllocOp {
    New,
    Delete,
    NewArray,
    DeleteArray,
    Malloc,
    Free,
};

struct AllocAction {
    AllocOp op;
    uint32_t tid;
    void *ptr;
    size_t size;
    size_t align;
    void *caller;
    uint64_t time;
};

constexpr const char *kAllocOpNames[] = {
    "New",
    "Delete",
    "NewArray",
    "DeleteArray",
    "Malloc",
    "Free",
};

constexpr bool kAllocOpIsAllocation[] = {
    true,
    false,
    true,
    false,
    true,
    false,
};

constexpr AllocOp kAllocOpPair[] = {
    AllocOp::Delete,
    AllocOp::New,
    AllocOp::DeleteArray,
    AllocOp::NewArray,
    AllocOp::Free,
    AllocOp::Malloc,
};

constexpr size_t kNone = (size_t)-1;
