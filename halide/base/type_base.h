#pragma once

// type handling code stripped from Halide runtime

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
// Forward declare type to allow naming typed handles.
// See Type.h for documentation.
template <typename T>
struct CPPHandleTraits;

/** Types in the halide type system. They can be ints, unsigned ints,
 * or floats (of various bit-widths), or a handle (which is always 64-bits).
 * Note that the int/uint/float values do not imply a specific bit width
 * (the bit width is expected to be encoded in a separate value).
 */
enum BaseTypeEnum : uint8_t {
    base_type_int = 0, //!< signed integers
    base_type_uint = 1, //!< unsigned integers
    base_type_float = 2, //!< floating point numbers
    base_type_handle = 3 //!< opaque pointer type (void *)
};

// Note that while __attribute__ can go before or after the declaration,
// __declspec apparently is only allowed before.
#ifndef IXGRAPH_ATTRIBUTE_ALIGN
#ifdef _MSC_VER
#define IXGRAPH_ATTRIBUTE_ALIGN(x) __declspec(align(x))
#else
#define IXGRAPH_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
#endif
#endif

/** A runtime tag for a type in the halide type system. Can be ints,
 * unsigned ints, or floats of various bit-widths (the 'bits'
 * field). Can also be vectors of the same (by setting the 'lanes'
 * field to something larger than one). This struct should be
 * exactly 32-bits in size. */
struct BaseType {
    /** The basic type code: signed integer, unsigned integer, or floating point. */
    IXGRAPH_ATTRIBUTE_ALIGN(1)
    BaseTypeEnum m_code; // BaseTypeEnum

    /** The number of bits of precision of a single scalar value of this type. */
    IXGRAPH_ATTRIBUTE_ALIGN(1)
    uint8_t m_bits;

    /** How many elements in a vector. This is 1 for scalar types. */
    IXGRAPH_ATTRIBUTE_ALIGN(2)
    uint16_t m_lanes;

    /** Construct a runtime representation of a Halide type from:
     * code: The fundamental type from an enum.
     * bits: The bit size of one element.
     * lanes: The number of vector elements in the type. */
    BaseType(BaseTypeEnum code, uint8_t bits, uint16_t lanes = 1)
        : m_code(code)
        , m_bits(bits)
        , m_lanes(lanes)
    {
    }

    /** Default constructor is required e.g. to declare halideir_trace_event
     * instances. */
    BaseType()
        : m_code((BaseTypeEnum)0)
        , m_bits(0)
        , m_lanes(0)
    {
    }

    /** Compare two types for equality. */
    bool operator==(const BaseType& other) const
    {
        return (m_code == other.m_code && m_bits == other.m_bits && m_lanes == other.m_lanes);
    }

    /** Size in bytes for a single element, even if width is not 1, of this type. */
    size_t bytes() const { return (m_bits + 7) / 8; }
};

namespace {

template <typename T>
struct BaseTypeHelper;

template <typename T>
struct BaseTypeHelper<T*> {
    operator BaseType()
    {
        return BaseType(base_type_handle, 64);
    }
};

template <typename T>
struct BaseTypeHelper<T&> {
    operator BaseType()
    {
        return BaseType(base_type_handle, 64);
    }
};

// Halide runtime does not require C++11
template <typename T>
struct BaseTypeHelper<T&&> {
    operator BaseType()
    {
        return BaseType(base_type_handle, 64);
    }
};

template <>
struct BaseTypeHelper<float> {
    operator BaseType() { return BaseType(base_type_float, 32); }
};

template <>
struct BaseTypeHelper<double> {
    operator BaseType() { return BaseType(base_type_float, 64); }
};

template <>
struct BaseTypeHelper<uint8_t> {
    operator BaseType() { return BaseType(base_type_uint, 8); }
};

template <>
struct BaseTypeHelper<uint16_t> {
    operator BaseType() { return BaseType(base_type_uint, 16); }
};

template <>
struct BaseTypeHelper<uint32_t> {
    operator BaseType() { return BaseType(base_type_uint, 32); }
};

template <>
struct BaseTypeHelper<uint64_t> {
    operator BaseType() { return BaseType(base_type_uint, 64); }
};

template <>
struct BaseTypeHelper<int8_t> {
    operator BaseType() { return BaseType(base_type_int, 8); }
};

template <>
struct BaseTypeHelper<int16_t> {
    operator BaseType() { return BaseType(base_type_int, 16); }
};

template <>
struct BaseTypeHelper<int32_t> {
    operator BaseType() { return BaseType(base_type_int, 32); }
};

template <>
struct BaseTypeHelper<int64_t> {
    operator BaseType() { return BaseType(base_type_int, 64); }
};

template <>
struct BaseTypeHelper<bool> {
    operator BaseType() { return BaseType(base_type_uint, 1); }
};

}

/** Construct the halide equivalent of a C type */
template <typename T>
BaseType BaseTypeCast()
{
    return BaseTypeHelper<T>();
}

// it is not necessary, and may produce warnings for some build configurations.
#ifdef _MSC_VER
#define HALIDEIR_ALWAYS_INLINE __forceinline
#else
#define HALIDEIR_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif
