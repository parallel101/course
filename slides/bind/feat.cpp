static constexpr struct change_these_options_to_select_what_will_be_printed
{
    constexpr static int longest_macro_name { 45 };
    constexpr static bool titles               = 1;
    constexpr static bool counters             = 1;
    constexpr static bool attributes           = 1;
    constexpr static bool standard_values      = 1;
    constexpr static bool compiler_specific    = 1;
    constexpr static bool core_features        = 1;
    constexpr static bool lib_features         = 1;
    constexpr static bool supported_features   = 1;
    constexpr static bool unsupported_features = 1;
    constexpr static bool sort_by_date         = 0;
    constexpr static bool separate_year_month  = 1;
    constexpr static bool separated_revisions  = 1;
    constexpr static bool latest_revisions     = 1;
    constexpr static bool cxx98                = 0;
    constexpr static bool cxx11                = 1;
    constexpr static bool cxx14                = 1;
    constexpr static bool cxx17                = 1;
    constexpr static bool cxx20                = 1;
    constexpr static bool cxx23                = 1;
    constexpr static bool cxx26                = 1;
    constexpr static bool cxx29                = 0;
} print;
 
#if __cplusplus < 201100
#  error "C++11 or better is required"
#endif
 
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
 
#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif
 
// Expect a string that starts with 6-decimal-digits or with '_' (if unsupported)
#define COMPILER_VALUE_INT(n) #n [0] == '_' ? 0 : \
    (#n[5] - '0') + (#n[4] - '0') * 10 + (#n[3] - '0') * 100 + \
    (#n[2] - '0') * 1000 + (#n[1] - '0') * 10000 + (#n[0] - '0') * 100000
#define COMPILER_FEATURE_ENTRY(expect, name) { #name, COMPILER_VALUE_INT(name), expect },
 
#if defined(__has_cpp_attribute) && defined(__GNUG__)
# define COMPILER_ATTRIBUTE(expect, name) { #name, __has_cpp_attribute(name), expect },
#else
# define COMPILER_ATTRIBUTE(expect, name) { #name, COMPILER_VALUE_INT(name), expect },
#endif
 
#define COMPILER_SPECIFIC_STRING(value) #value
#define COMPILER_SPECIFIC_ENTRY(name) { #name, COMPILER_SPECIFIC_STRING(name) },
 
class CompilerFeature
{
    char const* name_; long data_; long std_;
public:
    constexpr CompilerFeature(char const* name, long data, long std)
        : name_(name), data_(data), std_(std) {}
    constexpr CompilerFeature(CompilerFeature const&) = default;
    CompilerFeature& operator=(CompilerFeature const&) = default;
    bool operator<(CompilerFeature const& rhs) const
        { return std::strcmp(name_, rhs.name_) < 0; }
    bool operator==(CompilerFeature const& rhs) const
        { return std::strcmp(name_, rhs.name_) == 0; }
    constexpr bool supported() const { return data_ >= std_; }
    constexpr bool maybe() const { return data_ > 0; }
    constexpr char const* name() const { return name_; }
    constexpr long std() const { return std_; }
    constexpr long data() const { return data_; }
    void data(long x) { data_ = x; }
};
 
static /*constexpr*/ std::pair<const char*, const char*> compiler[] = {
    COMPILER_SPECIFIC_ENTRY(__cplusplus) //< not compiler specific, but useful :)
    COMPILER_SPECIFIC_ENTRY(__clang_major__)
    COMPILER_SPECIFIC_ENTRY(__clang_minor__)
    COMPILER_SPECIFIC_ENTRY(__clang_patchlevel__)
    COMPILER_SPECIFIC_ENTRY(__GNUG__)
    COMPILER_SPECIFIC_ENTRY(__GNUC_MINOR__)
    COMPILER_SPECIFIC_ENTRY(__GNUC_PATCHLEVEL__)
    // Add your favorite compiler specific macros. Undefined ones will not be printed.
};
 
static constexpr CompilerFeature cxx98_core[] = {
    COMPILER_FEATURE_ENTRY(199711L, __cpp_exceptions)
    COMPILER_FEATURE_ENTRY(199711L, __cpp_rtti)
};
 
static constexpr CompilerFeature cxx11_core[] = {
    COMPILER_FEATURE_ENTRY(200704L, __cpp_alias_templates)
    COMPILER_FEATURE_ENTRY(200809L, __cpp_attributes)
    COMPILER_FEATURE_ENTRY(200704L, __cpp_constexpr)
    COMPILER_FEATURE_ENTRY(201711L, __cpp_constexpr_in_decltype)
    COMPILER_FEATURE_ENTRY(200707L, __cpp_decltype)
    COMPILER_FEATURE_ENTRY(200604L, __cpp_delegating_constructors)
    COMPILER_FEATURE_ENTRY(201511L, __cpp_inheriting_constructors)
    COMPILER_FEATURE_ENTRY(200806L, __cpp_initializer_lists)
    COMPILER_FEATURE_ENTRY(200907L, __cpp_lambdas)
    COMPILER_FEATURE_ENTRY(200809L, __cpp_nsdmi)
    COMPILER_FEATURE_ENTRY(200907L, __cpp_range_based_for)
    COMPILER_FEATURE_ENTRY(200710L, __cpp_raw_strings)
    COMPILER_FEATURE_ENTRY(200710L, __cpp_ref_qualifiers)
    COMPILER_FEATURE_ENTRY(200610L, __cpp_rvalue_references)
    COMPILER_FEATURE_ENTRY(200410L, __cpp_static_assert)
    COMPILER_FEATURE_ENTRY(200806L, __cpp_threadsafe_static_init)
    COMPILER_FEATURE_ENTRY(200704L, __cpp_unicode_characters)
    COMPILER_FEATURE_ENTRY(200710L, __cpp_unicode_literals)
    COMPILER_FEATURE_ENTRY(200809L, __cpp_user_defined_literals)
    COMPILER_FEATURE_ENTRY(200704L, __cpp_variadic_templates)
};
 
static constexpr CompilerFeature cxx14_core[] = {
    COMPILER_FEATURE_ENTRY(201304L, __cpp_aggregate_nsdmi)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_binary_literals)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_constexpr)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_decltype_auto)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_generic_lambdas)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_init_captures)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_return_type_deduction)
    COMPILER_FEATURE_ENTRY(201309L, __cpp_sized_deallocation)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_variable_templates)
};
static constexpr CompilerFeature cxx14_lib[] = {
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_chrono_udls)
    COMPILER_FEATURE_ENTRY(201309L, __cpp_lib_complex_udls)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_exchange_function)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_generic_associative_lookup)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_integer_sequence)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_integral_constant_callable)
    COMPILER_FEATURE_ENTRY(201402L, __cpp_lib_is_final)
    COMPILER_FEATURE_ENTRY(201309L, __cpp_lib_is_null_pointer)
    COMPILER_FEATURE_ENTRY(201402L, __cpp_lib_make_reverse_iterator)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_make_unique)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_null_iterators)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_quoted_string_io)
    COMPILER_FEATURE_ENTRY(201210L, __cpp_lib_result_of_sfinae)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_robust_nonmodifying_seq_ops)
    COMPILER_FEATURE_ENTRY(201402L, __cpp_lib_shared_timed_mutex)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_string_udls)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_transformation_trait_aliases)
    COMPILER_FEATURE_ENTRY(201210L, __cpp_lib_transparent_operators)
    COMPILER_FEATURE_ENTRY(201402L, __cpp_lib_tuple_element_t)
    COMPILER_FEATURE_ENTRY(201304L, __cpp_lib_tuples_by_type)
};
 
static constexpr CompilerFeature cxx17_core[] = {
    COMPILER_FEATURE_ENTRY(201603L, __cpp_aggregate_bases)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_aligned_new)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_capture_star_this)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_constexpr)
    COMPILER_FEATURE_ENTRY(201703L, __cpp_deduction_guides)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_enumerator_attributes)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_fold_expressions)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_guaranteed_copy_elision)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_hex_float)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_if_constexpr)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_inline_variables)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_namespace_attributes)
    COMPILER_FEATURE_ENTRY(201510L, __cpp_noexcept_function_type)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_nontype_template_args)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_nontype_template_parameter_auto)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_range_based_for)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_static_assert)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_structured_bindings)
    COMPILER_FEATURE_ENTRY(201611L, __cpp_template_template_args)
    COMPILER_FEATURE_ENTRY(201611L, __cpp_variadic_using)
};
static constexpr CompilerFeature cxx17_lib[] = {
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_addressof_constexpr)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_lib_allocator_traits_is_always_equal)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_any)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_apply)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_array_constexpr)
    COMPILER_FEATURE_ENTRY(201510L, __cpp_lib_as_const)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_atomic_is_always_lock_free)
    COMPILER_FEATURE_ENTRY(201505L, __cpp_lib_bool_constant)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_boyer_moore_searcher)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_byte)
    COMPILER_FEATURE_ENTRY(201611L, __cpp_lib_chrono)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_clamp)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_enable_shared_from_this)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_execution)
    COMPILER_FEATURE_ENTRY(201703L, __cpp_lib_filesystem)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_gcd_lcm)
    COMPILER_FEATURE_ENTRY(201703L, __cpp_lib_hardware_interference_size)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_has_unique_object_representations)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_hypot)
    COMPILER_FEATURE_ENTRY(201505L, __cpp_lib_incomplete_container_elements)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_lib_invoke)
    COMPILER_FEATURE_ENTRY(201703L, __cpp_lib_is_aggregate)
    COMPILER_FEATURE_ENTRY(201703L, __cpp_lib_is_invocable)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_is_swappable)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_launder)
    COMPILER_FEATURE_ENTRY(201510L, __cpp_lib_logical_traits)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_make_from_tuple)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_lib_map_try_emplace)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_math_special_functions)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_memory_resource)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_node_extract)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_lib_nonmember_container_access)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_not_fn)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_optional)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_parallel_algorithm)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_raw_memory_algorithms)
    COMPILER_FEATURE_ENTRY(201603L, __cpp_lib_sample)
    COMPILER_FEATURE_ENTRY(201703L, __cpp_lib_scoped_lock)
    COMPILER_FEATURE_ENTRY(201505L, __cpp_lib_shared_mutex)
    COMPILER_FEATURE_ENTRY(201611L, __cpp_lib_shared_ptr_arrays)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_shared_ptr_weak_type)
    COMPILER_FEATURE_ENTRY(201606L, __cpp_lib_string_view)
    COMPILER_FEATURE_ENTRY(201611L, __cpp_lib_to_chars)
    COMPILER_FEATURE_ENTRY(201510L, __cpp_lib_transparent_operators)
    COMPILER_FEATURE_ENTRY(201510L, __cpp_lib_type_trait_variable_templates)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_lib_uncaught_exceptions)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_lib_unordered_map_try_emplace)
    COMPILER_FEATURE_ENTRY(202102L, __cpp_lib_variant)
    COMPILER_FEATURE_ENTRY(201411L, __cpp_lib_void_t)
};
 
static constexpr CompilerFeature cxx20_core[] = {
    COMPILER_FEATURE_ENTRY(201902L, __cpp_aggregate_paren_init)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_char8_t)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_concepts)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_conditional_explicit)
    COMPILER_FEATURE_ENTRY(202211L, __cpp_consteval)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_constexpr)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_constexpr_dynamic_alloc)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_constinit)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_deduction_guides)
    COMPILER_FEATURE_ENTRY(201707L, __cpp_designated_initializers)
    COMPILER_FEATURE_ENTRY(201707L, __cpp_generic_lambdas)
    COMPILER_FEATURE_ENTRY(201902L, __cpp_impl_coroutine)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_impl_destroying_delete)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_impl_three_way_comparison)
    COMPILER_FEATURE_ENTRY(201803L, __cpp_init_captures)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_modules)
    COMPILER_FEATURE_ENTRY(201911L, __cpp_nontype_template_args)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_using_enum)
};
static constexpr CompilerFeature cxx20_lib[] = {
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_array_constexpr)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_assume_aligned)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_atomic_flag_test)
    COMPILER_FEATURE_ENTRY(201711L, __cpp_lib_atomic_float)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_atomic_lock_free_type_aliases)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_atomic_ref)
    COMPILER_FEATURE_ENTRY(201711L, __cpp_lib_atomic_shared_ptr)
    COMPILER_FEATURE_ENTRY(201911L, __cpp_lib_atomic_value_initialization)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_atomic_wait)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_barrier)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_bind_front)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_bit_cast)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_bitops)
    COMPILER_FEATURE_ENTRY(201902L, __cpp_lib_bounded_array_traits)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_char8_t)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_chrono)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_lib_concepts)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_constexpr_algorithms)
    COMPILER_FEATURE_ENTRY(201711L, __cpp_lib_constexpr_complex)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_constexpr_dynamic_alloc)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_constexpr_functional)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_constexpr_iterator)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_constexpr_memory)
    COMPILER_FEATURE_ENTRY(201911L, __cpp_lib_constexpr_numeric)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_constexpr_string)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_constexpr_string_view)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_constexpr_tuple)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_constexpr_utility)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_constexpr_vector)
    COMPILER_FEATURE_ENTRY(201902L, __cpp_lib_coroutine)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_destroying_delete)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_endian)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_lib_erase_if)
    COMPILER_FEATURE_ENTRY(201902L, __cpp_lib_execution)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_format)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_generic_unordered_lookup)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_lib_int_pow2)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_lib_integer_comparison_functions)
    COMPILER_FEATURE_ENTRY(201902L, __cpp_lib_interpolate)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_is_constant_evaluated)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_is_layout_compatible)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_is_nothrow_convertible)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_is_pointer_interconvertible)
    COMPILER_FEATURE_ENTRY(201911L, __cpp_lib_jthread)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_latch)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_list_remove_return_type)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_math_constants)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_optional)
    COMPILER_FEATURE_ENTRY(201902L, __cpp_lib_polymorphic_allocator)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_ranges)
    COMPILER_FEATURE_ENTRY(201711L, __cpp_lib_remove_cvref)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_semaphore)
    COMPILER_FEATURE_ENTRY(201707L, __cpp_lib_shared_ptr_arrays)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_shift)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_lib_smart_ptr_for_overwrite)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_source_location)
    COMPILER_FEATURE_ENTRY(202002L, __cpp_lib_span)
    COMPILER_FEATURE_ENTRY(201902L, __cpp_lib_ssize)
    COMPILER_FEATURE_ENTRY(201711L, __cpp_lib_starts_ends_with)
    COMPILER_FEATURE_ENTRY(201803L, __cpp_lib_string_view)
    COMPILER_FEATURE_ENTRY(201803L, __cpp_lib_syncbuf)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_three_way_comparison)
    COMPILER_FEATURE_ENTRY(201711L, __cpp_lib_to_address)
    COMPILER_FEATURE_ENTRY(201907L, __cpp_lib_to_array)
    COMPILER_FEATURE_ENTRY(201806L, __cpp_lib_type_identity)
    COMPILER_FEATURE_ENTRY(201811L, __cpp_lib_unwrap_ref)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_variant)
};
 
static constexpr CompilerFeature cxx23_core[] = {
    COMPILER_FEATURE_ENTRY(202110L, __cpp_auto_cast)
    COMPILER_FEATURE_ENTRY(202211L, __cpp_constexpr)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_explicit_this_parameter)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_if_consteval)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_implicit_move)
    COMPILER_FEATURE_ENTRY(202211L, __cpp_multidimensional_subscript)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_named_character_escapes)
    COMPILER_FEATURE_ENTRY(202211L, __cpp_range_based_for)
    COMPILER_FEATURE_ENTRY(202011L, __cpp_size_t_suffix)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_static_call_operator)
};
static constexpr CompilerFeature cxx23_lib[] = {
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_adaptor_iterator_pair_constructor)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_algorithm_iterator_requirements)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_allocate_at_least)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_associative_heterogeneous_erasure)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_barrier)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_bind_back)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_byteswap)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_common_reference)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_common_reference_wrapper)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_concepts)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_constexpr_bitset)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_constexpr_charconv)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_constexpr_cmath)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_constexpr_memory)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_constexpr_typeinfo)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_containers_ranges)
    COMPILER_FEATURE_ENTRY(202211L, __cpp_lib_expected)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_flat_map)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_flat_set)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_format)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_format_ranges)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_formatters)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_forward_like)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_generator)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_invoke_r)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ios_noreplace)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_is_implicit_lifetime)
    COMPILER_FEATURE_ENTRY(202011L, __cpp_lib_is_scoped_enum)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_mdspan)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_modules)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_move_iterator_concept)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_move_only_function)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_optional)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_out_ptr)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_print)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_ranges)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_as_const)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_as_rvalue)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_cartesian_product)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_ranges_chunk)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_ranges_chunk_by)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_contains)
    COMPILER_FEATURE_ENTRY(202302L, __cpp_lib_ranges_enumerate)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_find_last)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_fold)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_ranges_iota)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_ranges_join_with)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_repeat)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_ranges_slide)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_ranges_starts_ends_with)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_ranges_stride)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_ranges_to_container)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_ranges_zip)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_reference_from_temporary)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_shift)
    COMPILER_FEATURE_ENTRY(202106L, __cpp_lib_spanstream)
    COMPILER_FEATURE_ENTRY(202011L, __cpp_lib_stacktrace)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_start_lifetime_as)
    COMPILER_FEATURE_ENTRY(202011L, __cpp_lib_stdatomic_h)
    COMPILER_FEATURE_ENTRY(202011L, __cpp_lib_string_contains)
    COMPILER_FEATURE_ENTRY(202110L, __cpp_lib_string_resize_and_overwrite)
    COMPILER_FEATURE_ENTRY(202102L, __cpp_lib_to_underlying)
    COMPILER_FEATURE_ENTRY(202207L, __cpp_lib_tuple_like)
    COMPILER_FEATURE_ENTRY(202202L, __cpp_lib_unreachable)
};
 
static constexpr CompilerFeature cxx26_core[] = {
    //< Continue to Populate
    COMPILER_FEATURE_ENTRY(202306L, __cpp_constexpr)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_deleted_function)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_pack_indexing)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_placeholder_variables)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_static_assert)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_structured_bindings)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_variadic_friend)
};
static constexpr CompilerFeature cxx26_lib[] = {
    //< Continue to Populate
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_algorithm_default_value_type)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_associative_heterogeneous_insertion)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_atomic_min_max)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_bind_back)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_bind_front)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_bitset)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_chrono)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_constexpr_algorithms)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_constexpr_cmath)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_constexpr_complex)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_constrained_equality)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_copyable_function)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_debugging)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_format)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_format_path)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_format_uchar)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_algorithm)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_array)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_char_traits)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_charconv)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_cstdlib)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_cstring)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_cwchar)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_errc)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_expected)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_feature_test_macros)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_functional)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_iterator)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_mdspan)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_memory)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_numeric)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_operator_new)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_optional)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_ranges)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_ratio)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_string_view)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_tuple)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_freestanding_utility)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_freestanding_variant)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_fstream_native_handle)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_function_ref)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_hazard_pointer)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_is_within_lifetime)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_linalg)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_not_fn)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_out_ptr)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_print)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_ranges_as_const)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_ranges_concat)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_ranges_generate_random)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_ratio)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_rcu)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_reference_wrapper)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_saturation_arithmetic)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_smart_ptr_owner_equality)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_span)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_span_initializer_list)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_sstream_from_string_view)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_string_view)
    COMPILER_FEATURE_ENTRY(202403L, __cpp_lib_submdspan)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_text_encoding)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_to_chars)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_to_string)
    COMPILER_FEATURE_ENTRY(202311L, __cpp_lib_tuple_like)
    COMPILER_FEATURE_ENTRY(202306L, __cpp_lib_variant)
};
 
static constexpr CompilerFeature cxx29_core[] = {
    //< Continue to Populate
    COMPILER_FEATURE_ENTRY(202604L, __cpp_core_TODO)
};
static constexpr CompilerFeature cxx29_lib[] = {
    //< Continue to Populate
    COMPILER_FEATURE_ENTRY(202604L, __cpp_lib_TODO)
};
 
static constexpr CompilerFeature attributes[] = {
    COMPILER_ATTRIBUTE(202207L, assume)
    COMPILER_ATTRIBUTE(200809L, carries_dependency)
    COMPILER_ATTRIBUTE(201309L, deprecated)
    COMPILER_ATTRIBUTE(201603L, fallthrough)
    COMPILER_ATTRIBUTE(202403L, indeterminate)
    COMPILER_ATTRIBUTE(201803L, likely)
    COMPILER_ATTRIBUTE(201603L, maybe_unused)
    COMPILER_ATTRIBUTE(201803L, no_unique_address)
    COMPILER_ATTRIBUTE(201907L, nodiscard)
    COMPILER_ATTRIBUTE(200809L, noreturn)
    COMPILER_ATTRIBUTE(201803L, unlikely)
};
 
inline void show_compiler_specific_info()
{
    std::printf("Compiler specific macros:\n");
    for (auto co : compiler)
        if (std::strcmp(co.first, co.second))
            std::printf("%*s %s\n", -print.longest_macro_name, co.first, co.second);
}
 
inline void print_compiler_feature(const CompilerFeature& x)
{
    if (not ((print.supported_features and x.maybe()) or
            (print.unsupported_features and not x.maybe())))
        return;
    auto print_year_month = [](long n)
    {
        return std::printf("%ld%s%02ld",
            n / 100, print.separate_year_month ? "-" : "", n % 100);
    };
    std::printf("%*s ", -print.longest_macro_name, x.name());
    x.maybe() ? print_year_month(x.data()) :
                    std::printf("------%s", print.separate_year_month ? "-" : "");
    if (print.standard_values)
        std::printf("  %c  ", (x.supported() ? (x.data() > x.std() ? '>' : '=') : '<')),
            print_year_month(x.std());
    std::puts("");
}
 
template<class Container>
inline void show(char const* const title, Container const& co)
{
    if (print.titles)
    {
        std::printf("%-s (", title);
        if (print.counters)
        {
            std::printf("%zd/", std::count_if(std::begin(co), std::end(co),
                [](CompilerFeature x)
                {
                    return x.supported();
                }));
        }
        std::printf("%td)\n", std::distance(std::begin(co), std::end(co)));
    }
    if (print.sort_by_date)
    {
        std::vector<CompilerFeature> v(std::begin(co), std::end(co));
        std::stable_sort(v.begin(), v.end(),
            [](CompilerFeature const& lhs, CompilerFeature const& rhs)
            {
                return lhs.data() < rhs.data();
            });
        std::for_each(v.cbegin(), v.cend(), print_compiler_feature);
    }
    else
        std::for_each(std::begin(co), std::end(co), print_compiler_feature);
    std::puts("");
}
 
inline void show_latest()
{
    auto latest_rev = []() -> int
    {
        return print.cxx29 ? 29 : print.cxx26 ? 26 : print.cxx23 ? 23 : print.cxx20 ? 20 :
               print.cxx17 ? 17 : print.cxx14 ? 14 : print.cxx11 ? 11 : 98;
    };
    std::vector<CompilerFeature> latest;
    auto add = [&latest](CompilerFeature x)
    {
        auto i = std::lower_bound(latest.begin(), latest.end(), x);
        if (i == latest.end() or not (*i == x))
            latest.insert(i, x);
        else if (i->data() < x.data())
            i->data(x.data());
    };
    char text[64];
    latest.reserve(512); // max macros
    if (print.core_features)
    {   // preserve reverse revision insertion order!
        if (print.cxx29) std::for_each(std::begin(cxx29_core), std::end(cxx29_core), add);
        if (print.cxx26) std::for_each(std::begin(cxx26_core), std::end(cxx26_core), add);
        if (print.cxx23) std::for_each(std::begin(cxx23_core), std::end(cxx23_core), add);
        if (print.cxx20) std::for_each(std::begin(cxx20_core), std::end(cxx20_core), add);
        if (print.cxx17) std::for_each(std::begin(cxx17_core), std::end(cxx17_core), add);
        if (print.cxx14) std::for_each(std::begin(cxx14_core), std::end(cxx14_core), add);
        if (print.cxx11) std::for_each(std::begin(cxx11_core), std::end(cxx11_core), add);
        if (print.cxx98) std::for_each(std::begin(cxx98_core), std::end(cxx98_core), add);
        std::snprintf(text, sizeof text, "ALL CORE MACROS UP TO C++%02i", latest_rev());
        show(text, latest);
    }
    latest.clear();
    if (print.lib_features)
    {   // preserve reverse revision insertion order!
        if (print.cxx29) std::for_each(std::begin(cxx29_lib), std::end(cxx29_lib), add);
        if (print.cxx26) std::for_each(std::begin(cxx26_lib), std::end(cxx26_lib), add);
        if (print.cxx23) std::for_each(std::begin(cxx23_lib), std::end(cxx23_lib), add);
        if (print.cxx20) std::for_each(std::begin(cxx20_lib), std::end(cxx20_lib), add);
        if (print.cxx17) std::for_each(std::begin(cxx17_lib), std::end(cxx17_lib), add);
        if (print.cxx14) std::for_each(std::begin(cxx14_lib), std::end(cxx14_lib), add);
        std::snprintf(text, sizeof text, "ALL LIB MACROS UP TO C++%02i", latest_rev());
        show(text, latest);
    }
}
 
int main()
{
    if (print.separated_revisions)
    {
        if (print.cxx98 and print.core_features) show("C++98 CORE", cxx98_core);
        if (print.cxx11 and print.core_features) show("C++11 CORE", cxx11_core);
        if (print.cxx14 and print.core_features) show("C++14 CORE", cxx14_core);
        if (print.cxx14 and print.lib_features ) show("C++14 LIB" , cxx14_lib);
        if (print.cxx17 and print.core_features) show("C++17 CORE", cxx17_core);
        if (print.cxx17 and print.lib_features ) show("C++17 LIB" , cxx17_lib);
        if (print.cxx20 and print.core_features) show("C++20 CORE", cxx20_core);
        if (print.cxx20 and print.lib_features ) show("C++20 LIB" , cxx20_lib);
        if (print.cxx23 and print.core_features) show("C++23 CORE", cxx23_core);
        if (print.cxx23 and print.lib_features ) show("C++23 LIB" , cxx23_lib);
        if (print.cxx26 and print.core_features) show("C++26 CORE", cxx26_core);
        if (print.cxx26 and print.lib_features ) show("C++26 LIB" , cxx26_lib);
        if (print.cxx29 and print.core_features) show("C++29 CORE", cxx29_core);
        if (print.cxx29 and print.lib_features ) show("C++29 LIB" , cxx29_lib);
    }
    if (print.latest_revisions) show_latest();
    if (print.attributes) show("ATTRIBUTES", attributes);
    if (print.compiler_specific) show_compiler_specific_info();
}
