#pragma once
//
// debug.hpp - prints everything!
//
// Source code available at: https://github.com/archibate/debug-hpp
//
// Usage:
//   debug(), "my variable is", your_variable;
//
// Results in:
//   your_file.cpp:233:  my variable is {1, 2, 3}
//
// (suppose your_variable is an std::vector)
//
// **WARNING**:
//   debug() only works in `Debug` build! It is automatically disabled in
//   `Release` build (we do this by checking whether the NDEBUG macro is
//   defined). Yeah, completely no outputs in `Release` build, this is by
//   design.
//
//   This is a feature for convenience, e.g. you don't have to remove all the
//   debug() sentences after debug done, simply switch to `Release` build and
//   everything debug is gone, no runtime overhead! And when you need debug
//   simply switch back to `Debug` build and everything debug() you written
//   before is back in life.
//
//   If you insist to use debug() even in `Release` build, please `#define
//   DEBUG_LEVEL 1` before including this header file.
//
//
// Assertion check is also supported:
//   debug().check(some_variable) > 0;
//
// Will trigger a 'trap' interrupt (__debugbreak for MSVC and __builtin_trap for
// GCC, configurable, see below) for the debugger to catch when `some_variable >
// 0` is false, as well as printing human readable error message:
//   your_file.cpp:233:  assertion failed: 3 < 0
//
//
// After debugging complete, no need to busy removing all debug() calls! Simply:
//   #define NDEBUG
// would supress all debug() prints and assertion checks, completely no runtime
// overhead. For CMake or Visual Studio users, simply switch to `Release` build
// would supress debug() prints. Since they automatically define `NDEBUG` for
// you in `Release`, `RelWithDebInfo` and `MinSizeRel` build types.
//
//
// TL;DR: This is a useful debugging utility the C++ programmers had all dreamed
// of:
//
//   1. print using the neat comma syntax, easy-to-use
//   2. supports printing STL objects including string, vector, tuple, optional,
//   variant, unique_ptr, type_info, error_code, and so on.
//   3. just add a member method named `repr`, e.g. `std::string repr() const {
//   ... }` to support printing your custom class!
//   4. classes that are not supported to print will be shown in something like
//   `[TypeName@0xdeadbeaf]` where 0xdeadbeaf is it's address.
//   5. highly configurable, customize the behaviour by defining the DEBUG_xxx
//   macros (see below)
//   6. when debug done, supress all debug messages by simply `#define NDEBUG`,
//   the whole library is disabled at compile-time, no runtime overhead
//   7. Thread safe, every line of message is always distinct, no annoying
//   interleaving output rushing into console (typical experience when using
//   cout)
//
//
// Here is a list of configurable macros, define them **before** including this
// header file to take effect:
//
// `#define DEBUG_LEVEL 0` (default when defined NDEBUG) - disable debug output,
// completely no runtime overhead
// `#define DEBUG_LEVEL 1` (default when !defined NDEBUG) - enable debug output,
// prints everything you asked to print
//
// `#define DEBUG_STEPPING 0` (default) - no step debugging
// `#define DEBUG_STEPPING 1` - enable step debugging, stops whenever debug
// output generated, manually press ENTER to continue
// `#define DEBUG_STEPPING 2` - enable step debugging, like 1, but trigger a
// 'trap' interrupt for debugger to catch instead
//
// `#define DEBUG_SHOW_TIMESTAMP 0` - do not print timestamp
// `#define DEBUG_SHOW_TIMESTAMP 1` - enable printing a timestamp for each line
// of debug output (e.g. "09:57:32")
// `#define DEBUG_SHOW_TIMESTAMP 2` (default) - printing timestamp relative to
// program staring time rather than system real time
//
// `#define DEBUG_SHOW_THREAD_ID 0` (default) - do not print the thread id
// `#define DEBUG_SHOW_THREAD_ID 1` - print the current thread id
//
// `#define DEBUG_SHOW_LOCATION 1` (default) - show source location mark before
// each line of the debug output (e.g. "file.cpp:233")
// `#define DEBUG_SHOW_LOCATION 0` - do not show the location mark
//
// `#define DEBUG_OUTPUT std::cerr <<` (default) - controls where to output the
// debug strings (must be callable as DEBUG_OUTPUT(str))
//
// `#define DEBUG_PANIC_METHOD 0` - throws an runtime error with debug string as
// message when assertion failed
// `#define DEBUG_PANIC_METHOD 1` (default) - print the error message when
// assertion failed, then triggers a 'trap' interrupt, useful for debuggers to
// catch, if no debuggers attached, the program would terminate
// `#define DEBUG_PANIC_METHOD 2` - print the error message when assertion
// failed, and then call std::terminate
// `#define DEBUG_PANIC_METHOD 3` - print the error message when assertion
// failed, do not terminate, do not throw any exception
//
// `#define DEBUG_REPR_NAME repr` (default) - if an object x have a member
// function like `x.repr()` or a global function `repr` supporting `repr(x)`,
// then the value of `x.repr()` or `repr(x)` will be printed instead for this
// object
//
// `#define DEBUG_RANGE_BRACE "{}"` (default) - controls format for range-like
// objects (supporting begin(x) and end(x)) in "{1, 2, 3, ...}"
// `#define DEBUG_RANGE_COMMA ", "` (default) - ditto
//
// `#define DEBUG_TUPLE_BRACE "{}"` (default) - controls format for tuple-like
// objects (supporting std::tuple_size<X>) in "{1, 2, 3}"
// `#define DEBUG_TUPLE_COMMA ", "` (default) - ditto
//
// `#define DEBUG_NAMED_MEMBER_MARK ": "` (default) - used in
// debug::named_member and DEBUG_REPR, e.g. '{name: "name", age: 42}'
//
// `#define DEBUG_MAGIC_ENUM magic_enum::enum_name` - enable printing enum in
// their name rather than value, if you have magic_enum.hpp
//
// `#define DEBUG_UNSIGNED_AS_HEXADECIMAL 0` (default) - print unsigned integers
// as decimal
// `#define DEBUG_UNSIGNED_AS_HEXADECIMAL 1` - print unsigned integers as
// hexadecimal
// `#define DEBUG_UNSIGNED_AS_HEXADECIMAL 2` - print unsigned integers as
// full-width hexadecimal
//
// `#define DEBUG_HEXADECIMAL_UPPERCASE 0` (default) - print hexadecimal values
// in lowercase (a-f)
// `#define DEBUG_HEXADECIMAL_UPPERCASE 1` - print hexadecimal values in
// uppercase (A-F)
//
// `#define DEBUG_SUPRESS_NON_ASCII 0` (default) - consider non-ascii characters
// in std::string as printable (e.g. UTF-8 encoded Chinese characters)
// `#define DEBUG_SUPRESS_NON_ASCII 1` - consider non-ascii characters in
// std::string as not printable (print them in e.g. '\xfe' instead)
//
// `#define DEBUG_SHOW_SOURCE_CODE_LINE 1` - enable debug output with detailed
// source code level information (requires readable source file path)
//
// `#define DEBUG_NULLOPT_STRING "nullopt"` (default) - controls how to print
// optional-like objects (supporting *x and (bool)x) when it is nullopt
//
// `#define DEBUG_NULLPTR_STRING "nullptr"` (default) - controls how to print
// pointer-like objects (supporting static_cast<void const volatile *>(x.get()))
// when it is nullptr
//
// `#define DEBUG_SMART_POINTER_MODE 1` (default) - print smart pointer as raw
// pointer address (e.g. 0xdeadbeaf)
// `#define DEBUG_SMART_POINTER_MODE 2` - print smart pointer as content value
// unless nullptr (e.g. 42 for std::shared_ptr<int>)
// `#define DEBUG_SMART_POINTER_MODE 3` - print smart pointer as both content
// value and raw pointer address (e.g. 42@0xdeadbeaf)
//
// `#define DEBUG_NAMESPACE_BEGIN` (default) - expose debug in the global
// namespace
// `#define DEBUG_NAMESPACE_END` (default) - ditto
//
// `#define DEBUG_NAMESPACE_BEGIN namespace mydebugger {` - expose debug in the
// the namespace `mydebugger`, and use it like `mydebugger::debug()`
// `#define DEBUG_NAMESPACE_END }` - ditto
//
// `#define DEBUG_CLASS_NAME debug` (default) - the default name for the debug
// class is `debug()`, you may define your custom name here
//
#ifndef DEBUG_LEVEL
# ifdef NDEBUG
#  define DEBUG_LEVEL 0
# else
#  define DEBUG_LEVEL 1
# endif
#endif
#ifndef DEBUG_SHOW_LOCATION
# define DEBUG_SHOW_LOCATION 1
#endif
#ifndef DEBUG_REPR_NAME
# define DEBUG_REPR_NAME repr
#endif
#ifndef DEBUG_FORMATTER_REPR_NAME
# define DEBUG_FORMATTER_REPR_NAME _debug_formatter_repr
#endif
#ifndef DEBUG_NAMESPACE_BEGIN
# define DEBUG_NAMESPACE_BEGIN
#endif
#ifndef DEBUG_NAMESPACE_END
# define DEBUG_NAMESPACE_END
#endif
#ifndef DEBUG_OUTPUT
# define DEBUG_OUTPUT std::cerr <<
#endif
#ifndef DEBUG_ENABLE_FILES_MATCH
# define DEBUG_ENABLE_FILES_MATCH 0
#endif
#ifndef DEBUG_ERROR_CODE_SHOW_NUMBER
# define DEBUG_ERROR_CODE_SHOW_NUMBER 0
#endif
#ifndef DEBUG_PANIC_METHOD
# define DEBUG_PANIC_METHOD 1
#endif
#ifndef DEBUG_STEPPING
# define DEBUG_STEPPING 0
#endif
#ifndef DEBUG_SUPRESS_NON_ASCII
# define DEBUG_SUPRESS_NON_ASCII 0
#endif
#ifndef DEBUG_SEPARATOR_FILE
# define DEBUG_SEPARATOR_FILE ':'
#endif
#ifndef DEBUG_SEPARATOR_LINE
# define DEBUG_SEPARATOR_LINE ':'
#endif
#ifndef DEBUG_SEPARATOR_TAB
# define DEBUG_SEPARATOR_TAB '\t'
#endif
#ifndef DEBUG_SOURCE_LINE_BRACE
# define DEBUG_SOURCE_LINE_BRACE "[]"
#endif
#ifndef DEBUG_TIMESTAMP_BRACE
# define DEBUG_TIMESTAMP_BRACE "[]"
#endif
#ifndef DEBUG_RANGE_BRACE
# define DEBUG_RANGE_BRACE "{}"
#endif
#ifndef DEBUG_RANGE_COMMA
# define DEBUG_RANGE_COMMA ", "
#endif
#ifndef DEBUG_TUPLE_BRACE
# define DEBUG_TUPLE_BRACE "{}"
#endif
#ifndef DEBUG_TUPLE_COMMA
# define DEBUG_TUPLE_COMMA ", "
#endif
#ifndef DEBUG_ENUM_BRACE
# define DEBUG_ENUM_BRACE "()"
#endif
#ifndef DEBUG_NAMED_MEMBER_MARK
# define DEBUG_NAMED_MEMBER_MARK ": "
#endif
#ifndef DEBUG_NULLOPT_STRING
# define DEBUG_NULLOPT_STRING "nullopt"
#endif
#ifndef DEBUG_NULLPTR_STRING
# define DEBUG_NULLPTR_STRING "nullptr"
#endif
#ifndef DEBUG_UNKNOWN_TYPE_BRACE
# define DEBUG_UNKNOWN_TYPE_BRACE "[]"
#endif
#ifndef DEBUG_ERROR_CODE_BRACE
# define DEBUG_ERROR_CODE_BRACE "[]"
#endif
#ifndef DEBUG_ERROR_CODE_INFIX
# define DEBUG_ERROR_CODE_INFIX ""
#endif
#ifndef DEBUG_ERROR_CODE_POSTFIX
# define DEBUG_ERROR_CODE_POSTFIX ": "
#endif
#ifndef DEBUG_ERROR_CODE_NO_ERROR
# define DEBUG_ERROR_CODE_NO_ERROR std::generic_category().message(0)
#endif
#ifndef DEBUG_UNKNOWN_TYPE_AT
# define DEBUG_UNKNOWN_TYPE_AT '@'
#endif
#ifndef DEBUG_SMART_POINTER_MODE
# define DEBUG_SMART_POINTER_MODE 1
#endif
#ifndef DEBUG_SMART_POINTER_AT
# define DEBUG_SMART_POINTER_AT '@'
#endif
#ifndef DEBUG_POINTER_HEXADECIMAL_PREFIX
# define DEBUG_POINTER_HEXADECIMAL_PREFIX "0x"
#endif
#ifndef DEBUG_UNSIGNED_AS_HEXADECIMAL
# define DEBUG_UNSIGNED_AS_HEXADECIMAL 0
#endif
#ifndef DEBUG_HEXADECIMAL_UPPERCASE
# define DEBUG_HEXADECIMAL_UPPERCASE 0
#endif
#ifndef DEBUG_SHOW_SOURCE_CODE_LINE
# define DEBUG_SHOW_SOURCE_CODE_LINE 0
#endif
#ifndef DEBUG_SHOW_TIMESTAMP
# define DEBUG_SHOW_TIMESTAMP 2
#endif
#ifdef DEBUG_CLASS_NAME
# define debug DEBUG_CLASS_NAME
#endif
#if DEBUG_LEVEL
# include <cstdint>
# include <cstdlib>
# include <iomanip>
# include <iostream>
# include <limits>
# include <system_error>
# if DEBUG_SHOW_SOURCE_CODE_LINE
#  include <fstream>
#  include <unordered_map>
# endif
# ifndef DEBUG_SOURCE_LOCATION
#  if __cplusplus >= 202002L
#   if defined(__has_include)
#    if __has_include(<source_location>)
#     include <source_location>
#     if __cpp_lib_source_location
#      define DEBUG_SOURCE_LOCATION std::source_location
#     endif
#    endif
#   endif
#  endif
# endif
# ifndef DEBUG_SOURCE_LOCATION
#  if __cplusplus >= 201505L
#   if defined(__has_include)
#    if __has_include(<experimental/source_location>)
#     include <experimental/source_location>
#     if __cpp_lib_experimental_source_location
#      define DEBUG_SOURCE_LOCATION std::experimental::source_location
#     endif
#    endif
#   endif
#  endif
# endif
# include <memory>
# include <sstream>
# include <string>
# include <type_traits>
# include <typeinfo>
# include <utility>
# ifndef DEBUG_CUSTOM_DEMANGLE
#  ifndef DEBUG_HAS_CXXABI_H
#   if defined(__has_include)
#    if defined(__unix__) && __has_include(<cxxabi.h>)
#     include <cxxabi.h>
#     define DEBUG_HAS_CXXABI_H
#    endif
#   endif
#  else
#   include <cxxabi.h>
#  endif
# endif
# if DEBUG_SHOW_TIMESTAMP == 1
#  if defined(__has_include)
#   if defined(__unix__) && __has_include(<sys/time.h>)
#    include <sys/time.h>
#    define DEBUG_HAS_SYS_TIME_H
#   endif
#  endif
#  ifndef DEBUG_HAS_SYS_TIME_H
#   include <chrono>
#  endif
# elif DEBUG_SHOW_TIMESTAMP == 2
#  include <chrono>
# endif
# if DEBUG_SHOW_THREAD_ID
#  include <thread>
# endif
# if defined(__has_include)
#  if __has_include(<variant>)
#   include <variant>
#  endif
# endif
# ifndef DEBUG_STRING_VIEW
#  if defined(__has_include)
#   if __cplusplus >= 201703L
#    if __has_include(<string_view>)
#     include <string_view>
#     if __cpp_lib_string_view
#      define DEBUG_STRING_VIEW std::string_view
#     endif
#    endif
#   endif
#  endif
# endif
# ifndef DEBUG_STRING_VIEW
#  include <string>
#  define DEBUG_STRING_VIEW std::string
# endif
# if __cplusplus >= 202002L
#  if defined(__has_cpp_attribute)
#   if __has_cpp_attribute(unlikely)
#    define DEBUG_UNLIKELY [[unlikely]]
#   else
#    define DEBUG_UNLIKELY
#   endif
#   if __has_cpp_attribute(likely)
#    define DEBUG_LIKELY [[likely]]
#   else
#    define DEBUG_LIKELY
#   endif
#   if __has_cpp_attribute(nodiscard)
#    define DEBUG_NODISCARD [[nodiscard]]
#   else
#    define DEBUG_NODISCARD
#   endif
#  else
#   define DEBUG_LIKELY
#   define DEBUG_UNLIKELY
#   define DEBUG_NODISCARD
#  endif
# else
#  define DEBUG_LIKELY
#  define DEBUG_UNLIKELY
#  define DEBUG_NODISCARD
# endif
# if DEBUG_STEPPING == 1
#  include <mutex>
#  if defined(__has_include)
#   if defined(__unix__)
#    if __has_include(<unistd.h>) && __has_include(<termios.h>)
#     include <termios.h>
#     include <unistd.h>
#     define DEBUG_STEPPING_HAS_TERMIOS 1
#    endif
#   endif
#   if defined(_WIN32)
#    if __has_include(<conio.h>)
#     include <conio.h>
#     define DEBUG_STEPPING_HAS_GETCH 1
#    endif
#   endif
#  endif
# endif
DEBUG_NAMESPACE_BEGIN

struct DEBUG_NODISCARD debug {
private:
# ifndef DEBUG_SOURCE_LOCATION
    struct debug_source_location {
        char const *fn;
        int ln;
        int col;
        char const *fun;

        char const *file_name() const noexcept {
            return fn;
        }

        int line() const noexcept {
            return ln;
        }

        int column() const noexcept {
            return col;
        }

        char const *function_name() const noexcept {
            return fun;
        }

        static debug_source_location current() noexcept {
            return {"???", 0, 0, "?"};
        }
    };
#  ifndef DEBUG_SOURCE_LOCATION_FAKER
#   define DEBUG_SOURCE_LOCATION_FAKER \
       { __FILE__, __LINE__, 0, __func__ }
#  endif
#  define DEBUG_SOURCE_LOCATION debug::debug_source_location
# endif
    template <class T>
    static auto debug_deref_avoid(T const &t)
        -> typename std::enable_if<!std::is_void<decltype(*t)>::value, decltype(*t)>::type {
        return *t;
    }

    struct debug_special_void {
        char const (&repr() const)[5] {
            return "void";
        }
    };

    template <class T>
    static auto debug_deref_avoid(T const &t)
        -> typename std::enable_if<std::is_void<decltype(*t)>::value, debug_special_void>::type {
        return debug_special_void();
    }

    static void debug_quotes(std::ostream &oss, DEBUG_STRING_VIEW sv,
                             char quote) {
        oss << quote;
        for (char c: sv) {
            switch (c) {
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            case '\\': oss << "\\\\"; break;
            case '\0': oss << "\\0"; break;
            default:
                if ((c >= 0 && c < 0x20) || c == 0x7F
# if DEBUG_SUPRESS_NON_ASCII
                    || (static_cast<unsigned char>(c) >= 0x80)
# endif
                ) {
                    auto f = oss.flags();
                    oss << "\\x" << std::hex << std::setfill('0')
                        << std::setw(2) << static_cast<int>(c)
# if DEBUG_HEXADECIMAL_UPPERCASE
                        << std::uppercase
# endif
                        ;
                    oss.flags(f);
                } else {
                    if (c == quote) {
                        oss << '\\';
                    }
                    oss << c;
                }
                break;
            }
        }
        oss << quote;
    }

    template <class T>
    struct debug_is_char_array : std::false_type {};

    template <std::size_t N>
    struct debug_is_char_array<char[N]> : std::true_type {};
# ifdef DEBUG_CUSTOM_DEMANGLE
    static std::string debug_demangle(char const *name) {
        return DEBUG_CUSTOM_DEMANGLE(name);
    }
# else
    static std::string debug_demangle(char const *name) {
#  ifdef DEBUG_HAS_CXXABI_H
        int status;
        char *p = abi::__cxa_demangle(name, 0, 0, &status);
        std::string s = p ? p : name;
        std::free(p);
#  else
        std::string s = name;
#  endif
        return s;
    }
# endif
public:
    struct debug_formatter {
        std::ostream &os;

        template <class T>
        debug_formatter &operator<<(T const &value) {
            debug_format(os, value);
            return *this;
        }
    };

private:
# if __cpp_if_constexpr && __cpp_concepts && \
     __cpp_lib_type_trait_variable_templates
public:
    template <class T, class U>
        requires std::is_same<T, U>::value
    static void debug_same_as(U &&) {}

private:
    template <class T>
    static void debug_format(std::ostream &oss, T const &t) {
        using std::begin;
        using std::end;
        if constexpr (debug_is_char_array<T>::value) {
            oss << t;
        } else if constexpr ((std::is_convertible<T,
                                                  DEBUG_STRING_VIEW>::value ||
                              std::is_convertible<T, std::string>::value)) {
            if constexpr (!std::is_convertible<T, DEBUG_STRING_VIEW>::value) {
                std::string s = t;
                debug_quotes(oss, s, '"');
            } else {
                debug_quotes(oss, t, '"');
            }
        } else if constexpr (std::is_same<T, bool>::value) {
            auto f = oss.flags();
            oss << std::boolalpha << t;
            oss.flags(f);
        } else if constexpr (std::is_same<T, char>::value ||
                             std::is_same<T, signed char>::value) {
            debug_quotes(oss, {reinterpret_cast<char const *>(&t), 1}, '\'');
        } else if constexpr (
#  if __cpp_char8_t
            std::is_same<T, char8_t>::value ||
#  endif
            std::is_same<T, char16_t>::value ||
            std::is_same<T, char32_t>::value ||
            std::is_same<T, wchar_t>::value) {
            auto f = oss.flags();
            oss << "'\\"
                << " xu U"[sizeof(T)] << std::hex << std::setfill('0')
                << std::setw(sizeof(T) * 2)
#  if DEBUG_HEXADECIMAL_UPPERCASE
                << std::uppercase
#  endif
                << static_cast<std::uint32_t>(t) << "'";
            oss.flags(f);
#  if DEBUG_UNSIGNED_AS_HEXADECIMAL
        } else if constexpr (std::is_integral<T>::value) {
            if constexpr (std::is_unsigned<T>::value) {
                auto f = oss.flags();
                oss << "0x" << std::hex << std::setfill('0')
#   if DEBUG_UNSIGNED_AS_HEXADECIMAL >= 2
                    << std::setw(sizeof(T) * 2)
#   endif
#   if DEBUG_HEXADECIMAL_UPPERCASE
                    << std::uppercase
#   endif
                    ;
                if constexpr (sizeof(T) == 1) {
                    oss << static_cast<unsigned int>(t);
                } else {
                    oss << t;
                }
                oss.flags(f);
            } else {
                oss << static_cast<std::uint64_t>(
                    static_cast<typename std::make_unsigned<T>::type>(t));
            }
#  else
        } else if constexpr (std::is_integral<T>::value) {
            oss << t; // static_cast<std::uint64_t>(static_cast<typename std::make_unsigned<T>::type>(t));
#  endif
        } else if constexpr (std::is_floating_point<T>::value) {
            auto f = oss.flags();
            oss << std::fixed
                << std::setprecision(std::numeric_limits<T>::digits10) << t;
            oss.flags(f);
        } else if constexpr (requires(T const &t) {
                                 static_cast<void const volatile *>(t.get());
                             }) {
            auto const *p = t.get();
            if (p != nullptr) {
#  if DEBUG_SMART_POINTER_MODE == 1
                debug_format(oss, *p);
#  elif DEBUG_SMART_POINTER_MODE == 2
                auto f = oss.flags();
                oss << DEBUG_POINTER_HEXADECIMAL_PREFIX << std::hex
                    << std::setfill('0')
#   if DEBUG_HEXADECIMAL_UPPERCASE
                    << std::uppercase
#   endif
                    ;
                oss << reinterpret_cast<std::uintptr_t>(
                    static_cast<void const volatile *>(p));
                oss.flags(f);
#  else
                debug_format(oss, *p);
                oss << DEBUG_SMART_POINTER_AT;
                auto f = oss.flags();
                oss << DEBUG_POINTER_HEXADECIMAL_PREFIX << std::hex
                    << std::setfill('0')
#   if DEBUG_HEXADECIMAL_UPPERCASE
                    << std::uppercase
#   endif
                    ;
                oss << reinterpret_cast<std::uintptr_t>(
                    static_cast<void const volatile *>(p));
                oss.flags(f);
#  endif
            } else {
                oss << DEBUG_NULLPTR_STRING;
            }
        } else if constexpr (std::is_same<T, std::errc>::value) {
            oss << DEBUG_ERROR_CODE_BRACE[0];
            if (t != std::errc()) {
                oss << std::generic_category().name() << DEBUG_ERROR_CODE_INFIX
#  if DEBUG_ERROR_CODE_SHOW_NUMBER
                    << ' ' << static_cast<int>(t)
#  endif
                    << DEBUG_ERROR_CODE_POSTFIX;
                oss << std::generic_category().message(static_cast<int>(t));
            } else {
                oss << DEBUG_ERROR_CODE_NO_ERROR;
            }
            oss << DEBUG_ERROR_CODE_BRACE[1];
        } else if constexpr (std::is_same<T, std::error_code>::value ||
                             std::is_same<T, std::error_condition>::value) {
            oss << DEBUG_ERROR_CODE_BRACE[0];
            if (t) {
                oss << t.category().name() << DEBUG_ERROR_CODE_INFIX
#  if DEBUG_ERROR_CODE_SHOW_NUMBER
                    << ' ' << t.value()
#  endif
                    << DEBUG_ERROR_CODE_POSTFIX << t.message();
            } else {
                oss << DEBUG_ERROR_CODE_NO_ERROR;
            }
            oss << DEBUG_ERROR_CODE_BRACE[1];
        } else if constexpr (requires(T const &t) {
                                 debug_same_as<typename T::type &>(t.get());
                             }) {
            debug_format(oss, t.get());
        } else if constexpr (requires(std::ostream &oss, T const &t) {
                                 oss << t;
                             }) {
            oss << t;
        } else if constexpr (std::is_pointer<T>::value ||
                             std::is_same<T, std::nullptr_t>::value) {
            if (t == nullptr) {
                auto f = oss.flags();
                oss << DEBUG_POINTER_HEXADECIMAL_PREFIX << std::hex
                    << std::setfill('0')
#  if DEBUG_HEXADECIMAL_UPPERCASE
                    << std::uppercase
#  endif
                    ;
                oss << reinterpret_cast<std::uintptr_t>(
                    reinterpret_cast<void const volatile *>(t));
                oss.flags(f);
            } else {
                oss << DEBUG_NULLPTR_STRING;
            }
        } else if constexpr (requires(T const &t) { begin(t) != end(t); }) {
            oss << DEBUG_RANGE_BRACE[0];
            bool add_comma = false;
            for (auto &&i: t) {
                if (add_comma) {
                    oss << DEBUG_RANGE_COMMA;
                }
                add_comma = true;
                debug_format(oss, std::forward<decltype(i)>(i));
            }
            oss << DEBUG_RANGE_BRACE[1];
        } else if constexpr (requires { std::tuple_size<T>::value; }) {
            oss << DEBUG_TUPLE_BRACE[0];
            bool add_comma = false;
            std::apply(
                [&](auto &&...args) {
                    (([&] {
                         if (add_comma) {
                             oss << DEBUG_TUPLE_COMMA;
                         }
                         add_comma = true;
                         debug_format(oss, std::forward<decltype(args)>(args));
                     }()),
                     ...);
                },
                t);
            oss << DEBUG_TUPLE_BRACE[1];
        } else if constexpr (std::is_enum<T>::value) {
#  ifdef DEBUG_MAGIC_ENUM
            oss << DEBUG_MAGIC_ENUM(t);
#  else
            oss << debug_demangle(typeid(T).name()) << DEBUG_ENUM_BRACE[0];
            oss << static_cast<typename std::underlying_type<T>::type>(t);
            oss << DEBUG_ENUM_BRACE[1];
#  endif
        } else if constexpr (std::is_same<T, std::type_info>::value) {
            oss << debug_demangle(t.name());
        } else if constexpr (requires(T const &t) { t.DEBUG_REPR_NAME(); }) {
            debug_format(oss, raw_repr_if_string(t.DEBUG_REPR_NAME()));
        } else if constexpr (requires(T const &t) { t.DEBUG_REPR_NAME(oss); }) {
            t.DEBUG_REPR_NAME(oss);
        } else if constexpr (requires(T const &t) { DEBUG_REPR_NAME(t); }) {
            debug_format(oss, raw_repr_if_string(DEBUG_REPR_NAME(t)));
        } else if constexpr (requires(debug_formatter const &out, T const &t) {
                                 t.DEBUG_FORMATTER_REPR_NAME(out);
                             }) {
            t.DEBUG_FORMATTER_REPR_NAME(debug_formatter{oss});
        } else if constexpr (requires(debug_formatter const &out, T const &t) {
                                 DEBUG_FORMATTER_REPR_NAME(out, t);
                             }) {
            DEBUG_FORMATTER_REPR_NAME(debug_formatter{oss}, t);
#  if __cpp_lib_variant
        } else if constexpr (requires { std::variant_size<T>::value; }) {
            visit([&oss](auto const &t) { debug_format(oss, t); }, t);
#  endif
        } else if constexpr (requires(T const &t) {
                                 (void)(*t);
                                 (void)(bool)t;
                             }) {
            if ((bool)t) {
                debug_format(oss, debug_deref_avoid(t));
            } else {
                oss << DEBUG_NULLOPT_STRING;
            }
        } else {
            oss << DEBUG_UNKNOWN_TYPE_BRACE[0]
                << debug_demangle(typeid(t).name()) << DEBUG_UNKNOWN_TYPE_AT;
            debug_format(oss,
                         reinterpret_cast<void const *>(std::addressof(t)));
            oss << DEBUG_UNKNOWN_TYPE_BRACE[1];
        }
    }
# else
    template <class T>
    struct debug_void {
        using type = void;
    };

    template <bool v>
    struct debug_bool_constant {
        enum {
            value = v
        };
    };

#  define DEBUG_COND(n, ...) \
      template <class T, class = void> \
      struct debug_cond_##n : std::false_type {}; \
      template <class T> \
      struct debug_cond_##n<T, \
                            typename debug_void<decltype(__VA_ARGS__)>::type> \
          : std::true_type {};
    DEBUG_COND(is_ostream_ok, std::declval<std::ostream &>()
                                  << std::declval<T const &>());
    DEBUG_COND(is_range, begin(std::declval<T const &>()) !=
                             end(std::declval<T const &>()));
    DEBUG_COND(is_tuple, std::tuple_size<T>::value);
    DEBUG_COND(is_member_repr, std::declval<T const &>().DEBUG_REPR_NAME());
    DEBUG_COND(is_member_repr_stream, std::declval<T const &>().DEBUG_REPR_NAME(
                                          std::declval<std::ostream &>()));
    DEBUG_COND(is_adl_repr, DEBUG_REPR_NAME(std::declval<T const &>()));
    DEBUG_COND(is_adl_repr_stream,
               DEBUG_REPR_NAME(std::declval<std::ostream &>(),
                               std::declval<T const &>()));
    DEBUG_COND(is_member_repr_debug,
               std::declval<T const &>().DEBUG_FORMATTER_REPR_NAME(
                   std::declval<debug_formatter const &>()));
    DEBUG_COND(is_adl_repr_debug, DEBUG_FORMATTER_REPR_NAME(
                                      std::declval<debug_formatter const &>(),
                                      std::declval<T const &>()));

    struct variant_test_lambda {
        std::ostream &oss;

        template <class T>
        void operator()(T const &) const {}
    };

#  if __cpp_lib_variant
    DEBUG_COND(is_variant, std::variant_size<T>::value);
#   else
    template <class>
    struct debug_cond_is_variant : std::false_type {};
#  endif
    DEBUG_COND(is_smart_pointer, static_cast<void const volatile *>(
                                     std::declval<T const &>().get()));
    DEBUG_COND(is_optional, (((void)*std::declval<T const &>(), (void)0),
                             ((void)(bool)std::declval<T const &>(), (void)0)));
    DEBUG_COND(
        reference_wrapper,
        (typename std::enable_if<
            std::is_same<typename T::type &,
                         decltype(std::declval<T const &>().get())>::value,
            int>::type)0);
#  define DEBUG_CON(n, ...) \
      template <class T> \
      struct debug_cond_##n : debug_bool_constant<__VA_ARGS__> {};
    DEBUG_CON(string, std::is_convertible<T, DEBUG_STRING_VIEW>::value ||
                          std::is_convertible<T, std::string>::value);
    DEBUG_CON(bool, std::is_same<T, bool>::value);
    DEBUG_CON(char, std::is_same<T, char>::value ||
                        std::is_same<T, signed char>::value);
    DEBUG_CON(error_code, std::is_same<T, std::errc>::value ||
                              std::is_same<T, std::error_code>::value ||
                              std::is_same<T, std::error_condition>::value);
#  if __cpp_char8_t
    DEBUG_CON(unicode_char, std::is_same<T, char8_t>::value ||
                                std::is_same<T, char16_t>::value ||
                                std::is_same<T, char32_t>::value ||
                                std::is_same<T, wchar_t>::value);
#  else
    DEBUG_CON(unicode_char, std::is_same<T, char16_t>::value ||
                                std::is_same<T, char32_t>::value ||
                                std::is_same<T, wchar_t>::value);
#  endif
#  if DEBUG_UNSIGNED_AS_HEXADECIMAL
    DEBUG_CON(integral_unsigned,
              std::is_integral<T>::value &&std::is_unsigned<T>::value);
#  else
    DEBUG_CON(integral_unsigned, false);
#  endif
    DEBUG_CON(integral, std::is_integral<T>::value);
    DEBUG_CON(floating_point, std::is_floating_point<T>::value);
    DEBUG_CON(pointer, std::is_pointer<T>::value ||
                           std::is_same<T, std::nullptr_t>::value);
    DEBUG_CON(enum, std::is_enum<T>::value);
    template <class T, class = void>
    struct debug_format_trait;

    template <class T>
    static void debug_format(std::ostream &oss, T const &t) {
        debug_format_trait<T>()(oss, t);
    }

    template <class T, class>
    struct debug_format_trait {
        void operator()(std::ostream &oss, T const &t) const {
            oss << DEBUG_UNKNOWN_TYPE_BRACE[0]
                << debug_demangle(typeid(t).name()) << DEBUG_UNKNOWN_TYPE_AT;
            debug_format(oss,
                         reinterpret_cast<void const *>(std::addressof(t)));
            oss << DEBUG_UNKNOWN_TYPE_BRACE[1];
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<debug_is_char_array<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            oss << t;
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               debug_cond_string<T>::value &&
               !std::is_convertible<T, DEBUG_STRING_VIEW>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            std::string s = t;
            debug_quotes(oss, s, '"');
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && debug_cond_string<T>::value &&
               std::is_convertible<T, DEBUG_STRING_VIEW>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            debug_quotes(oss, t, '"');
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<!debug_is_char_array<T>::value &&
                                   !debug_cond_string<T>::value &&
                                   debug_cond_bool<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            auto f = oss.flags();
            oss << std::boolalpha << t;
            oss.flags(f);
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && debug_cond_char<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            debug_quotes(oss, {reinterpret_cast<char const *>(&t), 1}, '\'');
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               debug_cond_unicode_char<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            auto f = oss.flags();
            oss << "'\\"
                << " xu U"[sizeof(T)] << std::hex << std::setfill('0')
                << std::setw(sizeof(T) * 2)
#  if DEBUG_HEXADECIMAL_UPPERCASE
                << std::uppercase
#  endif
                << static_cast<std::uint32_t>(t) << "'";
            oss.flags(f);
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               debug_cond_integral_unsigned<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            auto f = oss.flags();
            oss << "0x" << std::hex << std::setfill('0')
#  if DEBUG_UNSIGNED_AS_HEXADECIMAL >= 2
                << std::setw(sizeof(T) * 2)
#  endif
#  if DEBUG_HEXADECIMAL_UPPERCASE
                << std::uppercase
#  endif
                ;
            oss << static_cast<std::uint64_t>(
                static_cast<typename std::make_unsigned<T>::type>(t));
            oss.flags(f);
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               !debug_cond_integral_unsigned<T>::value &&
               debug_cond_integral<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            oss << t; // static_cast<std::uint64_t>(static_cast<typename std::make_unsigned<T>::type>(t));
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               !debug_cond_integral_unsigned<T>::value &&
               !debug_cond_integral<T>::value &&
               debug_cond_floating_point<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            auto f = oss.flags();
            oss << std::fixed
                << std::setprecision(std::numeric_limits<T>::digits10) << t;
            oss.flags(f);
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               !debug_cond_integral_unsigned<T>::value &&
               !debug_cond_integral<T>::value &&
               !debug_cond_floating_point<T>::value &&
               debug_cond_is_smart_pointer<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            auto const *p = t.get();
            if (p != nullptr) {
#  if DEBUG_SMART_POINTER_MODE == 1
                debug_format(oss, *p);
#  elif DEBUG_SMART_POINTER_MODE == 2
                auto f = oss.flags();
                oss << DEBUG_POINTER_HEXADECIMAL_PREFIX << std::hex
                    << std::setfill('0')
#   if DEBUG_HEXADECIMAL_UPPERCASE
                    << std::uppercase
#   endif
                    ;
                oss << reinterpret_cast<std::uintptr_t>(
                    static_cast<void const volatile *>(p));
                oss.flags(f);
#  else
                debug_format(oss, *p);
                oss << DEBUG_SMART_POINTER_AT;
                auto f = oss.flags();
                oss << DEBUG_POINTER_HEXADECIMAL_PREFIX << std::hex
                    << std::setfill('0')
#   if DEBUG_HEXADECIMAL_UPPERCASE
                    << std::uppercase
#   endif
                    ;
                oss << reinterpret_cast<std::uintptr_t>(
                    static_cast<void const volatile *>(p));
                oss.flags(f);
#  endif
            } else {
                oss << DEBUG_NULLPTR_STRING;
            }
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               !debug_cond_integral_unsigned<T>::value &&
               !debug_cond_integral<T>::value &&
               !debug_cond_floating_point<T>::value &&
               !debug_cond_is_smart_pointer<T>::value &&
               !debug_cond_error_code<T>::value &&
               debug_cond_reference_wrapper<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            debug_format(oss, t.get());
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               !debug_cond_integral_unsigned<T>::value &&
               !debug_cond_integral<T>::value &&
               !debug_cond_floating_point<T>::value &&
               !debug_cond_is_smart_pointer<T>::value &&
               !debug_cond_error_code<T>::value &&
               !debug_cond_reference_wrapper<T>::value &&
               debug_cond_is_ostream_ok<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            oss << t;
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               !debug_cond_integral_unsigned<T>::value &&
               !debug_cond_integral<T>::value &&
               !debug_cond_floating_point<T>::value &&
               !debug_cond_is_smart_pointer<T>::value &&
               !debug_cond_error_code<T>::value &&
               !debug_cond_reference_wrapper<T>::value &&
               !debug_cond_is_ostream_ok<T>::value &&
               debug_cond_pointer<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            auto f = oss.flags();
            if (t == nullptr) {
                oss << DEBUG_POINTER_HEXADECIMAL_PREFIX << std::hex
                    << std::setfill('0')
#  if DEBUG_HEXADECIMAL_UPPERCASE
                    << std::uppercase
#  endif
                    ;
                oss << reinterpret_cast<std::uintptr_t>(
                    reinterpret_cast<void const volatile *>(t));
                oss.flags(f);
            } else {
                oss << DEBUG_NULLPTR_STRING;
            }
        }
    };

    template <class T>
    struct debug_format_trait<
        T, typename std::enable_if<
               !debug_is_char_array<T>::value && !debug_cond_string<T>::value &&
               !debug_cond_bool<T>::value && !debug_cond_char<T>::value &&
               !debug_cond_unicode_char<T>::value &&
               !debug_cond_integral_unsigned<T>::value &&
               !debug_cond_integral<T>::value &&
               !debug_cond_floating_point<T>::value &&
               !debug_cond_is_smart_pointer<T>::value &&
               !debug_cond_error_code<T>::value &&
               !debug_cond_reference_wrapper<T>::value &&
               !debug_cond_is_ostream_ok<T>::value &&
               !debug_cond_pointer<T>::value &&
               debug_cond_is_range<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            oss << DEBUG_RANGE_BRACE[0];
            bool add_comma = false;
            auto b = begin(t);
            auto e = end(t);
            for (auto it = b; it != e; ++it) {
                if (add_comma) {
                    oss << DEBUG_RANGE_COMMA;
                }
                add_comma = true;
                debug_format(oss, std::forward<decltype(*it)>(*it));
            }
            oss << DEBUG_RANGE_BRACE[1];
        }
    };
#  if __cpp_lib_integer_sequence
    template <class F, class Tuple, std::size_t... I>
    static void debug_apply_impl(F &&f, Tuple &&t, std::index_sequence<I...>) {
        std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
    }

    template <class F, class Tuple, std::size_t... I>
    static void debug_apply(F &&f, Tuple &&t) {
        debug_apply_impl(
            std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<
                std::tuple_size<typename std::decay<Tuple>::type>::value>{});
    }
#  else
    template <std::size_t... I>
    struct debug_index_sequence {};

    template <std::size_t N, std::size_t... I>
    struct debug_make_index_sequence
        : debug_make_index_sequence<N - 1, I..., N - 1> {};

    template <std::size_t... I>
    struct debug_make_index_sequence<0, I...> : debug_index_sequence<I...> {};

    template <class F, class Tuple, std::size_t... I>
    static void debug_apply_impl(F &&f, Tuple &&t, debug_index_sequence<I...>) {
        return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
    }

    template <class F, class Tuple, std::size_t... I>
    static void debug_apply(F &&f, Tuple &&t) {
        return debug_apply_impl(
            std::forward<F>(f), std::forward<Tuple>(t),
            debug_make_index_sequence<
                std::tuple_size<typename std::decay<Tuple>::type>::value>{});
    }
#  endif
    struct debug_apply_lambda {
        std::ostream &oss;
        bool &add_comma;

        template <class Arg>
        void call(Arg &&arg) const {
            if (add_comma) {
                oss << DEBUG_TUPLE_COMMA;
            }
            add_comma = true;
            debug_format(oss, std::forward<decltype(arg)>(arg));
        }

        template <class... Args>
        void operator()(Args &&...args) const {
            int unused[] = {(call<Args>(std::forward<Args>(args)), 0)...};
            (void)unused;
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            debug_cond_is_tuple<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            oss << DEBUG_TUPLE_BRACE[0];
            bool add_comma = false;
            debug_apply(debug_apply_lambda{oss, add_comma}, t);
            oss << DEBUG_TUPLE_BRACE[1];
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value &&
            debug_cond_enum<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
#  ifdef DEBUG_MAGIC_ENUM
            oss << DEBUG_MAGIC_ENUM(t);
#  else
            oss << debug_demangle(typeid(T).name()) << DEBUG_ENUM_BRACE[0];
            oss << static_cast<typename std::underlying_type<T>::type>(t);
            oss << DEBUG_ENUM_BRACE[1];
#  endif
        }
    };

    template <class V>
    struct debug_format_trait<std::type_info, V> {
        void operator()(std::ostream &oss, std::type_info const &t) const {
            oss << debug_demangle(t.name());
        }
    };

    template <class V>
    struct debug_format_trait<std::errc, V> {
        void operator()(std::ostream &oss, std::errc const &t) const {
            oss << DEBUG_ERROR_CODE_BRACE[0];
            if (t != std::errc()) {
                oss << std::generic_category().name() << DEBUG_ERROR_CODE_INFIX
#  if DEBUG_ERROR_CODE_SHOW_NUMBER
                    << ' ' << static_cast<int>(t)
#  endif
                    << DEBUG_ERROR_CODE_POSTFIX;
                oss << std::generic_category().message(static_cast<int>(t));
            } else {
                oss << DEBUG_ERROR_CODE_NO_ERROR;
            }
            oss << DEBUG_ERROR_CODE_BRACE[1];
        }
    };

    template <class V>
    struct debug_format_trait<std::error_code, V> {
        void operator()(std::ostream &oss, std::error_code const &t) const {
            oss << DEBUG_ERROR_CODE_BRACE[0];
            if (t) {
                oss << t.category().name() << DEBUG_ERROR_CODE_INFIX
#  if DEBUG_ERROR_CODE_SHOW_NUMBER
                    << ' ' << t.value()
#  endif
                    << DEBUG_ERROR_CODE_POSTFIX << t.message();
            } else {
                oss << DEBUG_ERROR_CODE_NO_ERROR;
            }
            oss << DEBUG_ERROR_CODE_BRACE[1];
        }
    };

    template <class V>
    struct debug_format_trait<std::error_condition, V> {
        void operator()(std::ostream &oss,
                        std::error_condition const &t) const {
            oss << DEBUG_UNKNOWN_TYPE_BRACE[0];
            if (t) {
                oss << t.category().name() << " error " << t.value() << ": "
                    << t.message();
            } else {
                oss << "no error";
            }
            oss << DEBUG_UNKNOWN_TYPE_BRACE[1];
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            debug_cond_is_member_repr<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            debug_format(oss, raw_repr_if_string(t.DEBUG_REPR_NAME()));
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            !debug_cond_is_member_repr<T>::value &&
            debug_cond_is_member_repr_stream<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            t.DEBUG_REPR_NAME(oss);
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            !debug_cond_is_member_repr<T>::value &&
            !debug_cond_is_member_repr_stream<T>::value &&
            debug_cond_is_adl_repr<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            debug_format(oss, raw_repr_if_string(DEBUG_REPR_NAME(t)));
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            !debug_cond_is_member_repr<T>::value &&
            !debug_cond_is_member_repr_stream<T>::value &&
            !debug_cond_is_adl_repr<T>::value &&
            debug_cond_is_adl_repr_stream<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            DEBUG_REPR_NAME(oss, t);
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            !debug_cond_is_member_repr<T>::value &&
            !debug_cond_is_member_repr_stream<T>::value &&
            !debug_cond_is_adl_repr<T>::value &&
            !debug_cond_is_adl_repr_stream<T>::value &&
            debug_cond_is_member_repr_debug<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            t.DEBUG_FORMATTER_REPR_NAME(debug_formatter{oss});
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            !debug_cond_is_member_repr<T>::value &&
            !debug_cond_is_member_repr_stream<T>::value &&
            !debug_cond_is_adl_repr<T>::value &&
            !debug_cond_is_adl_repr_stream<T>::value &&
            !debug_cond_is_member_repr_debug<T>::value &&
            debug_cond_is_adl_repr_debug<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            DEBUG_FORMATTER_REPR_NAME(debug_formatter{oss}, t);
        }
    };

    struct debug_visit_lambda {
        std::ostream &oss;

        template <class T>
        void operator()(T const &t) const {
            debug_format(oss, t);
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            !debug_cond_is_member_repr<T>::value &&
            !debug_cond_is_member_repr_stream<T>::value &&
            !debug_cond_is_adl_repr<T>::value &&
            !debug_cond_is_adl_repr_stream<T>::value &&
            debug_cond_is_variant<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            visit(debug_visit_lambda{oss}, t);
        }
    };

    template <class T>
    struct debug_format_trait<
        T,
        typename std::enable_if<
            !debug_cond_string<T>::value && !debug_cond_bool<T>::value &&
            !debug_cond_char<T>::value && !debug_cond_unicode_char<T>::value &&
            !debug_cond_integral_unsigned<T>::value &&
            !debug_cond_integral<T>::value &&
            !debug_cond_floating_point<T>::value &&
            !debug_cond_is_smart_pointer<T>::value &&
            !debug_cond_error_code<T>::value &&
            !debug_cond_reference_wrapper<T>::value &&
            !debug_cond_is_ostream_ok<T>::value &&
            !debug_cond_pointer<T>::value && !debug_cond_is_range<T>::value &&
            !debug_cond_is_tuple<T>::value && !debug_cond_enum<T>::value &&
            !debug_cond_is_member_repr<T>::value &&
            !debug_cond_is_member_repr_stream<T>::value &&
            !debug_cond_is_adl_repr<T>::value &&
            !debug_cond_is_adl_repr_stream<T>::value &&
            !debug_cond_is_variant<T>::value &&
            debug_cond_is_optional<T>::value>::type> {
        void operator()(std::ostream &oss, T const &t) const {
            if ((bool)t) {
                debug_format(oss, debug_deref_avoid(t));
            } else {
                oss << DEBUG_NULLOPT_STRING;
            }
        }
    };
# endif
    std::ostringstream oss;

    enum {
        silent = 0,
        print = 1,
        panic = 2,
        supress = 3,
    } state;

    DEBUG_SOURCE_LOCATION loc;
# if DEBUG_SHOW_TIMESTAMP == 2
#  if __cpp_inline_variables
    static inline std::chrono::steady_clock::time_point const tp0 =
        std::chrono::steady_clock::now();
#  endif
# endif
    debug &add_location_marks() {
# if DEBUG_SHOW_TIMESTAMP == 1
#  ifdef DEBUG_HAS_SYS_TIME_H
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        std::tm now = *std::localtime(&tv.tv_sec);
        oss << std::put_time(&now, "%H:%M:%S.");
        auto flags = oss.flags();
        oss << std::setw(3) << std::setfill('0');
        oss << (tv.tv_usec / 1000) % 1000;
        oss.flags(flags);
#  else
        auto tp = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(tp);
        std::tm now = *std::gmtime(&t);
        oss << std::put_time(&now, "%H:%M:%S.");
        auto flags = oss.flags();
        oss << std::setw(3) << std::setfill('0');
        oss << std::chrono::duration_cast<std::chrono::milliseconds>(
                   tp.time_since_epoch())
                       .count() %
                   1000;
        oss.flags(flags);
#  endif
        oss << ' ';
# elif DEBUG_SHOW_TIMESTAMP == 2
#  if __cpp_inline_variables
        auto dur = std::chrono::steady_clock::now() - tp0;
#  else
        static std::chrono::steady_clock::time_point const tp0 =
            std::chrono::steady_clock::now();
        auto dur = std::chrono::steady_clock::now() - tp0;
#  endif
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        auto flags = oss.flags();
        oss << std::setw(3) << std::setfill(' ');
        oss << (elapsed / 1000) % 1000;
        oss << '.';
        oss << std::setw(3) << std::setfill('0');
        oss << elapsed % 1000;
        oss.flags(flags);
        oss << ' ';
# endif
# if DEBUG_SHOW_THREAD_ID
        oss << '[' << std::this_thread::get_id() << ']' << ' ';
# endif
        char const *fn = loc.file_name();
        for (char const *fp = fn; *fp; ++fp) {
            if (*fp == '/') {
                fn = fp + 1;
            }
        }
# if DEBUG_SHOW_LOCATION
        oss << fn << DEBUG_SEPARATOR_FILE << loc.line() << DEBUG_SEPARATOR_LINE
            << DEBUG_SEPARATOR_TAB;
# endif
# if DEBUG_SHOW_SOURCE_CODE_LINE
        {
            static thread_local std::unordered_map<std::string, std::string>
                fileCache;
            auto key = std::to_string(loc.line()) + loc.file_name();
            if (auto it = fileCache.find(key);
                it != fileCache.end() && !it->second.empty()) {
                DEBUG_LIKELY {
                    oss << DEBUG_SOURCE_LINE_BRACE[0] << it->second
                        << DEBUG_SOURCE_LINE_BRACE[1];
                }
            } else if (auto file = std::ifstream(loc.file_name());
                       file.is_open()) {
                DEBUG_LIKELY {
                    std::string line;
                    for (std::uint32_t i = 0; i < loc.line(); ++i) {
                        if (!std::getline(file, line)) {
                            DEBUG_UNLIKELY {
                                line.clear();
                                break;
                            }
                        }
                    }
                    if (auto pos = line.find_first_not_of(" \t\r\n");
                        pos != line.npos) {
                        DEBUG_LIKELY {
                            line = line.substr(pos);
                        }
                    }
                    if (!line.empty()) {
                        DEBUG_LIKELY {
                            if (line.back() == ';') {
                                DEBUG_LIKELY {
                                    line.pop_back();
                                }
                            }
                            oss << DEBUG_SOURCE_LINE_BRACE[0] << line
                                << DEBUG_SOURCE_LINE_BRACE[1];
                        }
                    }
                    fileCache.try_emplace(key, std::move(line));
                }
            } else {
                oss << DEBUG_SOURCE_LINE_BRACE[0] << '?'
                    << DEBUG_SOURCE_LINE_BRACE[1];
                fileCache.try_emplace(key);
            }
        }
# endif
        oss << ' ';
        return *this;
    }

    template <class T>
    struct DEBUG_NODISCARD debug_condition {
    private:
        debug &d;
        T const &t;

        template <class U>
        debug &check(bool cond, U const &u, char const *sym) {
            if (!cond) {
                DEBUG_UNLIKELY {
                    d.on_error("assertion failed:") << t << sym << u;
                }
            }
            return d;
        }

    public:
        explicit debug_condition(debug &d, T const &t) noexcept : d(d), t(t) {}

        template <class U>
        debug &operator<(U const &u) {
            return check(t < u, u, "<");
        }

        template <class U>
        debug &operator>(U const &u) {
            return check(t > u, u, ">");
        }

        template <class U>
        debug &operator<=(U const &u) {
            return check(t <= u, u, "<=");
        }

        template <class U>
        debug &operator>=(U const &u) {
            return check(t >= u, u, ">=");
        }

        template <class U>
        debug &operator==(U const &u) {
            return check(t == u, u, "==");
        }

        template <class U>
        debug &operator!=(U const &u) {
            return check(t != u, u, "!=");
        }
    };

    debug &on_error(char const *msg) {
        if (state != supress) {
            state = panic;
            add_location_marks();
        } else {
            oss << ' ';
        }
        oss << msg;
        return *this;
    }

    template <class T>
    debug &on_print(T const &t) {
        if (state == supress) {
            return *this;
        }
        if (state == silent) {
            state = print;
            add_location_marks();
        } else {
            oss << ' ';
        }
        debug_format(oss, t);
        return *this;
    }
# if DEBUG_ENABLE_FILES_MATCH
    static bool file_detected(char const *file) noexcept {
        static auto files = std::getenv("DEBUG_FILES");
        if (!files) {
            return true;
        }
        DEBUG_STRING_VIEW sv = files;
        /* std::size_t pos = 0, nextpos; */
        /* while ((nextpos = sv.find(' ')) != sv.npos) { */
        /*     if (sv.substr(pos, nextpos - pos) == tag) { */
        /*         return true; */
        /*     } */
        /*     pos = nextpos + 1; */
        /* } */
        if (sv.find(file) != sv.npos) {
            return true;
        }
        return false;
    }
# endif
public:
    explicit debug(bool enable = true,
                   DEBUG_SOURCE_LOCATION const &loc =
                       DEBUG_SOURCE_LOCATION::current()) noexcept
        : state(enable
# if DEBUG_ENABLE_FILES_MATCH
                        && file_detected(loc.file_name())
# endif
                    ? silent
                    : supress),
          loc(loc) {
    }

    debug &setloc(DEBUG_SOURCE_LOCATION const &newloc =
                      DEBUG_SOURCE_LOCATION::current()) noexcept {
        loc = newloc;
        return *this;
    }

    debug &noloc() noexcept {
        if (state == silent) {
            state = print;
        }
        return *this;
    }

    debug(debug &&) = delete;
    debug(debug const &) = delete;

    template <class T>
    debug_condition<T> check(T const &t) noexcept {
        return debug_condition<T>{*this, t};
    }

    template <class T>
    debug_condition<T> operator>>(T const &t) noexcept {
        return debug_condition<T>{*this, t};
    }

    debug &fail(bool fail = true) {
        if (fail) {
            DEBUG_UNLIKELY {
                on_error("failed:");
            }
        } else {
            state = supress;
        }
        return *this;
    }

    debug &on(bool enable) {
        if (!enable) {
            DEBUG_LIKELY {
                state = supress;
            }
        }
        return *this;
    }

    template <class T>
    debug &operator<<(T const &t) {
        return on_print(t);
    }

    template <class T>
    debug &operator,(T const &t) {
        return on_print(t);
    }

    ~debug()
# if DEBUG_PANIC_METHOD == 0
        noexcept(false)
# endif
    {
        if (state == panic) {
            DEBUG_UNLIKELY {
# if DEBUG_PANIC_METHOD == 0
                throw std::runtime_error(oss.str());
# elif DEBUG_PANIC_METHOD == 1
                oss << '\n';
                DEBUG_OUTPUT(oss.str());
#  if defined(DEBUG_PANIC_CUSTOM_TRAP)
                DEBUG_PANIC_CUSTOM_TRAP;
                return;
#  elif defined(_MSC_VER)
                __debugbreak();
                return;
#  elif defined(__GNUC__) && defined(__has_builtin)
#   if __has_builtin(__builtin_trap)
                __builtin_trap();
                return;
#   else
                std::terminate();
#   endif
#  else
                std::terminate();
#  endif
# elif DEBUG_PANIC_METHOD == 2
                oss << '\n';
                DEBUG_OUTPUT(oss.str());
                std::terminate();
# else
                oss << '\n';
                DEBUG_OUTPUT(oss.str());
                return;
# endif
            }
        }
        if (state == print) {
            oss << '\n';
            DEBUG_OUTPUT(oss.str());
        }
# if DEBUG_STEPPING == 1
        static std::mutex mutex;
        std::lock_guard lock(mutex);
#  ifdef DEBUG_CUSTOM_STEPPING
        DEBUG_CUSTOM_STEPPING(msg);
#  elif DEBUG_STEPPING_HAS_TERMIOS
        struct termios tc, oldtc;
        bool tty = isatty(0);
        if (tty) {
            tcgetattr(0, &oldtc);
            tc = oldtc;
            tc.c_lflag &= ~ICANON;
            tc.c_lflag &= ~ECHO;
            tcsetattr(0, TCSANOW, &tc);
        }
        std::cerr << "--More--" << std::flush;
        static char buf[100];
        read(0, buf, sizeof buf);
        std::cerr << "\r        \r";
        if (tty) {
            tcsetattr(0, TCSANOW, &oldtc);
        }
#  elif DEBUG_STEPPING_HAS_GETCH
        std::cerr << "--More--" << std::flush;
        getch();
        std::cerr << "\r        \r";
#  else
        std::cerr << "[Press ENTER to continue]" << std::flush;
        std::string line;
        std::getline(std::cin, line);
#  endif
# elif DEBUG_STEPPING == 2
#  if defined(DEBUG_PANIC_CUSTOM_TRAP)
        DEBUG_PANIC_CUSTOM_TRAP;
        return;
#  elif defined(_MSC_VER)
        __debugbreak();
#  elif defined(__GNUC__) && defined(__has_builtin)
#   if __has_builtin(__builtin_trap)
        __builtin_trap();
#   else
        std::terminate();
#   endif
#  endif
# endif
    }

    operator std::string() {
        std::string ret = oss.str();
        state = supress;
        return ret;
    }

    template <class T>
    struct named_member_t {
        char const *name;
        T const &value;

        void DEBUG_REPR_NAME(std::ostream &os) const {
            os << name << DEBUG_NAMED_MEMBER_MARK;
            debug_format(os, value);
        }
    };

    template <class T>
    static named_member_t<T> named_member(char const *name, T const &value) {
        return {name, value};
    }

    template <class T>
    struct raw_repr_t {
        T const &value;

        void DEBUG_REPR_NAME(std::ostream &os) const {
            os << value;
        }
    };

    template <class T>
    static raw_repr_t<T> raw_repr(T const &value) {
        return {value};
    }

    template <class T, typename std::enable_if<
                           std::is_convertible<T, std::string>::value ||
                               std::is_convertible<T, DEBUG_STRING_VIEW>::value,
                           int>::type = 0>
    static raw_repr_t<T> raw_repr_if_string(T const &value) {
        return {value};
    }

    template <class T, typename std::enable_if<
                           !(std::is_convertible<T, std::string>::value ||
                             std::is_convertible<T, DEBUG_STRING_VIEW>::value),
                           int>::type = 0>
    static T const &raw_repr_if_string(T const &value) {
        return value;
    }
};
# if defined(_MSC_VER) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL)
#  define DEBUG_REPR(...) \
      __pragma( \
          message("Please turn on /Zc:preprocessor before using DEBUG_REPR!"))
#  define DEBUG_REPR_GLOBAL(...) \
      __pragma( \
          message("Please turn on /Zc:preprocessor before using DEBUG_REPR!"))
#  define DEBUG_REPR_GLOBAL_TEMPLATED(...) \
      __pragma( \
          message("Please turn on /Zc:preprocessor before using DEBUG_REPR!"))
#  define DEBUG_PP_VA_OPT_SUPPORT(...) 0
# else
#  define DEBUG_PP_CONCAT_(a, b)                             a##b
#  define DEBUG_PP_CONCAT(a, b)                              DEBUG_PP_CONCAT_(a, b)
#  define DEBUG_PP_GET_1(a, ...)                             a
#  define DEBUG_PP_GET_2(a, b, ...)                          b
#  define DEBUG_PP_GET_3(a, b, c, ...)                       c
#  define DEBUG_PP_GET_4(a, b, c, d, ...)                    d
#  define DEBUG_PP_GET_5(a, b, c, d, e, ...)                 e
#  define DEBUG_PP_GET_6(a, b, c, d, e, f, ...)              f
#  define DEBUG_PP_GET_7(a, b, c, d, e, f, g, ...)           g
#  define DEBUG_PP_GET_8(a, b, c, d, e, f, g, h, ...)        h
#  define DEBUG_PP_GET_9(a, b, c, d, e, f, g, h, i, ...)     i
#  define DEBUG_PP_GET_10(a, b, c, d, e, f, g, h, i, j, ...) j
#  define DEBUG_PP_VA_EMPTY_(...)                            DEBUG_PP_GET_2(__VA_OPT__(, ) 0, 1, )
#  define DEBUG_PP_VA_OPT_SUPPORT                            !DEBUG_PP_VA_EMPTY_
#  if DEBUG_PP_VA_OPT_SUPPORT(?)
#   define DEBUG_PP_VA_EMPTY(...) DEBUG_PP_VA_EMPTY_(__VA_ARGS__)
#  else
#   define DEBUG_PP_VA_EMPTY(...) 0
#  endif
#  define DEBUG_PP_IF(a, t, f)     DEBUG_PP_IF_(a, t, f)
#  define DEBUG_PP_IF_(a, t, f)    DEBUG_PP_IF__(a, t, f)
#  define DEBUG_PP_IF__(a, t, f)   DEBUG_PP_IF___(DEBUG_PP_VA_EMPTY a, t, f)
#  define DEBUG_PP_IF___(a, t, f)  DEBUG_PP_IF____(a, t, f)
#  define DEBUG_PP_IF____(a, t, f) DEBUG_PP_IF_##a(t, f)
#  define DEBUG_PP_IF_0(t, f)      DEBUG_PP_UNWRAP_BRACE(f)
#  define DEBUG_PP_IF_1(t, f)      DEBUG_PP_UNWRAP_BRACE(t)
#  define DEBUG_PP_NARG(...) \
      DEBUG_PP_IF((__VA_ARGS__), (0), \
                  (DEBUG_PP_NARG_(__VA_ARGS__, 26, 25, 24, 23, 22, 21, 20, 19, \
                                  18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, \
                                  6, 5, 4, 3, 2, 1)))
#  define DEBUG_PP_NARG_(...) DEBUG_PP_NARG__(__VA_ARGS__)
#  define DEBUG_PP_NARG__(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, \
                          _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, \
                          _23, _24, _25, _26, N, ...) \
      N
#  define DEBUG_PP_FOREACH(f, ...) \
      DEBUG_PP_FOREACH_(DEBUG_PP_NARG(__VA_ARGS__), f, __VA_ARGS__)
#  define DEBUG_PP_FOREACH_(N, f, ...)  DEBUG_PP_FOREACH__(N, f, __VA_ARGS__)
#  define DEBUG_PP_FOREACH__(N, f, ...) DEBUG_PP_FOREACH_##N(f, __VA_ARGS__)
#  define DEBUG_PP_FOREACH_0(f, ...)
#  define DEBUG_PP_FOREACH_1(f, a)                f(a)
#  define DEBUG_PP_FOREACH_2(f, a, b)             f(a) f(b)
#  define DEBUG_PP_FOREACH_3(f, a, b, c)          f(a) f(b) f(c)
#  define DEBUG_PP_FOREACH_4(f, a, b, c, d)       f(a) f(b) f(c) f(d)
#  define DEBUG_PP_FOREACH_5(f, a, b, c, d, e)    f(a) f(b) f(c) f(d) f(e)
#  define DEBUG_PP_FOREACH_6(f, a, b, c, d, e, g) f(a) f(b) f(c) f(d) f(e) f(g)
#  define DEBUG_PP_FOREACH_7(f, a, b, c, d, e, g, h) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h)
#  define DEBUG_PP_FOREACH_8(f, a, b, c, d, e, g, h, i) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i)
#  define DEBUG_PP_FOREACH_9(f, a, b, c, d, e, g, h, i, j) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j)
#  define DEBUG_PP_FOREACH_10(f, a, b, c, d, e, g, h, i, j, k) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k)
#  define DEBUG_PP_FOREACH_11(f, a, b, c, d, e, g, h, i, j, k, l) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l)
#  define DEBUG_PP_FOREACH_12(f, a, b, c, d, e, g, h, i, j, k, l, m) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m)
#  define DEBUG_PP_FOREACH_13(f, a, b, c, d, e, g, h, i, j, k, l, m, n) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n)
#  define DEBUG_PP_FOREACH_14(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o)
#  define DEBUG_PP_FOREACH_15(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) f(p)
#  define DEBUG_PP_FOREACH_16(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q)
#  define DEBUG_PP_FOREACH_17(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r)
#  define DEBUG_PP_FOREACH_18(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s)
#  define DEBUG_PP_FOREACH_19(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s, t) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s) f(t)
#  define DEBUG_PP_FOREACH_20(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s, t, u) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s) f(t) f(u)
#  define DEBUG_PP_FOREACH_21(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s, t, u, v) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s) f(t) f(u) f(v)
#  define DEBUG_PP_FOREACH_22(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s, t, u, v, w) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s) f(t) f(u) f(v) f(w)
#  define DEBUG_PP_FOREACH_23(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s, t, u, v, w, x) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s) f(t) f(u) f(v) f(w) f(x)
#  define DEBUG_PP_FOREACH_24(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s, t, u, v, w, x, y) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s) f(t) f(u) f(v) f(w) f(x) f(y)
#  define DEBUG_PP_FOREACH_25(f, a, b, c, d, e, g, h, i, j, k, l, m, n, o, p, \
                              q, r, s, t, u, v, w, x, y, z) \
      f(a) f(b) f(c) f(d) f(e) f(g) f(h) f(i) f(j) f(k) f(l) f(m) f(n) f(o) \
          f(p) f(q) f(r) f(s) f(t) f(u) f(v) f(w) f(x) f(y) f(z)
#  define DEBUG_PP_STRINGIFY(...)     DEBUG_PP_STRINGIFY(__VA_ARGS__)
#  define DEBUG_PP_STRINGIFY_(...)    #__VA_ARGS__
#  define DEBUG_PP_EXPAND(...)        DEBUG_PP_EXPAND_(__VA_ARGS__)
#  define DEBUG_PP_EXPAND_(...)       __VA_ARGS__
#  define DEBUG_PP_UNWRAP_BRACE(...)  DEBUG_PP_UNWRAP_BRACE_ __VA_ARGS__
#  define DEBUG_PP_UNWRAP_BRACE_(...) __VA_ARGS__
#  define DEBUG_REPR_ON_EACH(x) \
      if (add_comma) \
          formatter.os << DEBUG_TUPLE_COMMA; \
      else \
          add_comma = true; \
      formatter.os << #x DEBUG_NAMED_MEMBER_MARK; \
      formatter << x;
#  define DEBUG_REPR(...) \
      template <class debug_formatter> \
      void DEBUG_FORMATTER_REPR_NAME(debug_formatter formatter) const { \
          formatter.os << DEBUG_TUPLE_BRACE[0]; \
          bool add_comma = false; \
          DEBUG_PP_FOREACH(DEBUG_REPR_ON_EACH, __VA_ARGS__) \
          formatter.os << DEBUG_TUPLE_BRACE[1]; \
      }
#  define DEBUG_REPR_GLOBAL_ON_EACH(x) \
      if (add_comma) \
          formatter.os << DEBUG_TUPLE_COMMA; \
      else \
          add_comma = true; \
      formatter.os << #x DEBUG_NAMED_MEMBER_MARK; \
      formatter << object.x;
#  define DEBUG_REPR_GLOBAL(T, ...) \
      template <class debug_formatter> \
      void DEBUG_FORMATTER_REPR_NAME(debug_formatter formatter, \
                                     T const &object) { \
          formatter.os << DEBUG_TUPLE_BRACE[0]; \
          bool add_comma = false; \
          DEBUG_PP_FOREACH(DEBUG_REPR_GLOBAL_ON_EACH, __VA_ARGS__) \
          formatter.os << DEBUG_TUPLE_BRACE[1]; \
      }
#  define DEBUG_REPR_GLOBAL_TEMPLATED(T, Tmpls, TmplsClassed, ...) \
      template <class debug_formatter, DEBUG_PP_UNWRAP_BRACE(TmplsClassed)> \
      void DEBUG_FORMATTER_REPR_NAME( \
          debug_formatter formatter, \
          T<DEBUG_PP_UNWRAP_BRACE(Tmpls)> const &object) { \
          formatter.os << DEBUG_TUPLE_BRACE[0]; \
          bool add_comma = false; \
          DEBUG_PP_FOREACH(DEBUG_REPR_GLOBAL_ON_EACH, __VA_ARGS__) \
          formatter.os << DEBUG_TUPLE_BRACE[1]; \
      }
# endif
DEBUG_NAMESPACE_END
#else
# include <string>
DEBUG_NAMESPACE_BEGIN

struct debug {
    debug(bool = true, char const * = nullptr) noexcept {}

    debug(debug &&) = delete;
    debug(debug const &) = delete;

    template <class T>
    debug &operator,(T const &) {
        return *this;
    }

    template <class T>
    debug &operator<<(T const &) {
        return *this;
    }

    debug &on(bool) {
        return *this;
    }

    debug &fail(bool = true) {
        return *this;
    }

    ~debug() noexcept(false) {}

private:
    struct debug_condition {
        debug &d;

        explicit debug_condition(debug &d) : d(d) {}

        template <class U>
        debug &operator<(U const &) {
            return d;
        }

        template <class U>
        debug &operator>(U const &) {
            return d;
        }

        template <class U>
        debug &operator<=(U const &) {
            return d;
        }

        template <class U>
        debug &operator>=(U const &) {
            return d;
        }

        template <class U>
        debug &operator==(U const &) {
            return d;
        }

        template <class U>
        debug &operator!=(U const &) {
            return d;
        }
    };

public:
    template <class... Ts>
    debug &setloc(Ts &&...ts) noexcept {
        return *this;
    }

    debug &noloc() noexcept {
        return *this;
    }

    template <class T>
    debug_condition check(T const &) noexcept {
        return debug_condition{*this};
    }

    template <class T>
    debug_condition operator>>(T const &) noexcept {
        return debug_condition{*this};
    }

    operator std::string() {
        return {};
    }

    template <class T>
    struct named_member_t {
        char const *name;
        T const &value;
    };

    template <class T>
    static named_member_t<T> named_member(char const *name, T const &value) {
        return {name, value};
    }

    template <class T>
    struct raw_repr_t {
        T const &value;
    };

    template <class T>
    static raw_repr_t<T> raw_repr(T const &value) {
        return {value};
    }

    template <class T>
    static T raw_repr_if_string(T const &value) {
        return value;
    }
};

# define DEBUG_REPR(...)
DEBUG_NAMESPACE_END
#endif
#ifdef DEBUG_CLASS_NAME
# undef debug
#elif DEBUG_LEVEL
# ifdef DEBUG_SOURCE_LOCATION_FAKER
#  define debug() debug(true, DEBUG_SOURCE_LOCATION_FAKER)
# endif
#endif
