#pragma once

#include "error.h"
#include "float16.h"
#include "type_base.h"
#include "util.h"
#include <stdint.h>

/** \file
 * Defines halide types
 */

/** A set of types to represent a C++ function signature. This allows
 * two things.  First, proper prototypes can be provided for Halide
 * generated functions, giving better compile time type
 * checking. Second, C++ name mangling can be done to provide link
 * time type checking for both Halide generated functions and calls
 * from Halide to external functions.
 *
 * These are intended to be constexpr producable, but we don't depend
 * on C++11 yet. In C++14, it is possible these will be replaced with
 * introspection/reflection facilities.
 *
 * CPPHandleTraits has to go outside the Halide namespace due to template
 * resolution rules. TODO(zalman): Do all types need to be in global namespace?
 */
//@{

/** A structure to represent the (unscoped) name of a C++ composite type for use
 * as a single argument (or return value) in a function signature.
 *
 * Currently does not support the restrict qualifier, references, or
 * r-value references.  These features cannot be used in extern
 * function calls from Halide or in the generated function from
 * Halide, but their applicability seems limited anyway.
 */
struct CPPTypeName {
    /// An enum to indicate whether a C++ type is non-composite, a struct, class, or union
    enum CPPTypeType {
        Simple, ///< "int"
        Struct, ///< "struct Foo"
        Class, ///< "class Foo"
        Union, ///< "union Foo"
        Enum, ///< "enum Foo"
    } m_cpp_type_type; // Note: order is reflected in map_to_name table in CPlusPlusMangle.cpp

    std::string m_name;

    CPPTypeName(CPPTypeType cpp_type_type, const std::string& name)
        : m_cpp_type_type(cpp_type_type)
        , m_name(name)
    {
    }

    bool operator==(const CPPTypeName& rhs) const
    {
        return m_cpp_type_type == rhs.m_cpp_type_type && m_name == rhs.m_name;
    }

    bool operator!=(const CPPTypeName& rhs) const
    {
        return !(*this == rhs);
    }

    bool operator<(const CPPTypeName& rhs) const
    {
        return m_cpp_type_type < rhs.m_cpp_type_type || (m_cpp_type_type == rhs.m_cpp_type_type && m_name < rhs.m_name);
    }
};

/** A structure to represent the fully scoped name of a C++ composite
 * type for use in generating function signatures that use that type.
 *
 * This is intended to be a constexpr usable type, but we don't depend
 * on C++11 yet. In C++14, it is possible this will be replaced with
 * introspection/reflection facilities.
 */
struct CPPHandleTypeInfo {
    CPPTypeName m_inner_name;
    std::vector<std::string> m_namespaces;
    std::vector<CPPTypeName> m_enclosing_types;

    /// One set of modifiers on a type.
    /// The const/volatile/restrict propertises are "inside" the pointer property.
    enum Modifier : uint8_t {
        Const = 1 << 0, ///< Bitmask flag for "const"
        Volatile = 1 << 1, ///< Bitmask flag for "volatile"
        Restrict = 1 << 2, ///< Bitmask flag for "restrict"
        Pointer = 1 << 3, ///< Bitmask flag for a pointer "*"
    };
    std::vector<uint8_t> m_cpp_type_modifiers; /// Qualifiers and indirections on type. 0 is innermost.

    /// References are separate because they only occur at the outermost level.
    /// No modifiers are needed for references as they are not allowed to apply
    /// to the reference itself. (This isn't true for restrict, but that is a C++
    /// extension anyway.) If modifiers are needed, the last entry in the above
    /// array would be the modifers for the reference.
    enum ReferenceType : uint8_t {
        NotReference = 0,
        LValueReference = 1, // "&"
        RValueReference = 2, // "&&"
    };
    ReferenceType m_reference_type;

    CPPHandleTypeInfo(const CPPTypeName& inner_name,
        const std::vector<std::string>& namespaces = {},
        const std::vector<CPPTypeName>& enclosing_types = {},
        const std::vector<uint8_t>& modifiers = {},
        ReferenceType reference_type = NotReference)
        : m_inner_name(inner_name)
        , m_namespaces(namespaces)
        , m_enclosing_types(enclosing_types)
        , m_cpp_type_modifiers(modifiers)
        , m_reference_type(reference_type)
    {
    }
};
//@}

template <typename T>
struct CPPTypeToName {
    static const bool known_type = false;
};

#define DECLARE_EXTERN_TYPE(TypeType, Type)          \
    template <>                                      \
    struct CPPTypeToName<Type> {                     \
        static const bool known_type = true;         \
        static CPPTypeName name()                    \
        {                                            \
            return { CPPTypeName::TypeType, #Type }; \
        }                                            \
    }

#define DECLARE_EXTERN_SIMPLE_TYPE(T) DECLARE_EXTERN_TYPE(Simple, T)
#define DECLARE_EXTERN_STRUCT_TYPE(T) DECLARE_EXTERN_TYPE(Struct, T)
#define DECLARE_EXTERN_CLASS_TYPE(T) DECLARE_EXTERN_TYPE(Class, T)
#define DECLARE_EXTERN_UNION_TYPE(T) DECLARE_EXTERN_TYPE(Union, T)

DECLARE_EXTERN_SIMPLE_TYPE(bool);
DECLARE_EXTERN_SIMPLE_TYPE(int8_t);
DECLARE_EXTERN_SIMPLE_TYPE(uint8_t);
DECLARE_EXTERN_SIMPLE_TYPE(int16_t);
DECLARE_EXTERN_SIMPLE_TYPE(uint16_t);
DECLARE_EXTERN_SIMPLE_TYPE(int32_t);
DECLARE_EXTERN_SIMPLE_TYPE(uint32_t);
DECLARE_EXTERN_SIMPLE_TYPE(int64_t);
DECLARE_EXTERN_SIMPLE_TYPE(uint64_t);
DECLARE_EXTERN_SIMPLE_TYPE(float);
DECLARE_EXTERN_SIMPLE_TYPE(double);

// You can make arbitrary user-defined types be "Known" using the
// macro above. This is useful for making Param<> arguments for
// Generators type safe. e.g.,
//
//    struct MyFunStruct { ... };
//
//    ...
//
//    DECLARE_EXTERN_STRUCT_TYPE(MyFunStruct);
//
//    ...
//
//    class MyGenerator : public Generator<MyGenerator> {
//       Param<const MyFunStruct *> my_struct_ptr;
//       ...
//    };

// Default case (should be only Unknown types, since we specialize for Known types below).
// We require that all unknown types be pointers, and translate them all to void*
// (preserving const-ness and volatile-ness).
template <typename T, bool KnownType>
struct HandleTraitsHelper {
    static const CPPHandleTypeInfo* type_info(bool is_ptr,
        CPPHandleTypeInfo::ReferenceType ref_type)
    {
        static_assert(!KnownType, "Only unknown types handled here");
        internal_assert(is_ptr) << "Unknown types must be pointers";
        internal_assert(ref_type == CPPHandleTypeInfo::NotReference) << "Unknown types must not be references";
        static const CPPHandleTypeInfo the_info {
            { CPPTypeName::Simple, "void" },
            {},
            {},
            { (uint8_t)(CPPHandleTypeInfo::Pointer | (std::is_const<T>::value ? CPPHandleTypeInfo::Const : 0) | (std::is_volatile<T>::value ? CPPHandleTypeInfo::Volatile : 0)) },
            CPPHandleTypeInfo::NotReference
        };
        return &the_info;
    }
};

// Known types
template <typename T>
struct HandleTraitsHelper<T, true> {

    static const CPPHandleTypeInfo make_info(bool is_ptr,
        CPPHandleTypeInfo::ReferenceType ref_type)
    {
        CPPHandleTypeInfo the_info = {
            CPPTypeToName<typename std::remove_cv<T>::type>::name(),
            {},
            {},
            { (uint8_t)((is_ptr ? CPPHandleTypeInfo::Pointer : 0) | (std::is_const<T>::value ? CPPHandleTypeInfo::Const : 0) | (std::is_volatile<T>::value ? CPPHandleTypeInfo::Volatile : 0)) },
            ref_type
        };
        // Pull off any namespaces
        the_info.m_inner_name.m_name = HalideIR::Internal::extract_namespaces(the_info.m_inner_name.m_name,
            the_info.m_namespaces);
        return the_info;
    }

    static const CPPHandleTypeInfo* type_info(bool is_ptr,
        CPPHandleTypeInfo::ReferenceType ref_type)
    {
        static const CPPHandleTypeInfo the_info = make_info(is_ptr, ref_type);
        return &the_info;
    }
};

/** A type traits template to provide a CPPHandleTypeInfo
 * value from a C++ type.
 *
 * Note the type represented is implicitly a pointer.
 *
 * A NULL pointer of type CPPHandleTraits represents "void *".
 * This is chosen for compactness or representation as Type is a very
 * widely used data structure.
 */
template <typename T>
struct CPPHandleTraits {
    // NULL here means "void *". This trait must return a pointer to a
    // global structure. I.e. it should never be freed.
    inline static const CPPHandleTypeInfo* type_info() { return nullptr; }
};

template <typename T>
struct CPPHandleTraits<T*> {
    inline static const CPPHandleTypeInfo* type_info()
    {
        return HandleTraitsHelper<T, CPPTypeToName<typename std::remove_cv<T>::type>::known_type>::type_info(true, CPPHandleTypeInfo::NotReference);
    }
};

template <typename T>
struct CPPHandleTraits<T&> {
    inline static const CPPHandleTypeInfo* type_info()
    {
        return HandleTraitsHelper<T, CPPTypeToName<typename std::remove_cv<T>::type>::known_type>::type_info(false, CPPHandleTypeInfo::LValueReference);
    }
};

template <typename T>
struct CPPHandleTraits<T&&> {
    inline static const CPPHandleTypeInfo* type_info()
    {
        return HandleTraitsHelper<T, CPPTypeToName<typename std::remove_cv<T>::type>::known_type>::type_info(false, CPPHandleTypeInfo::RValueReference);
    }
};

template <>
struct CPPHandleTraits<const char*> {
    inline static const CPPHandleTypeInfo* type_info()
    {
        static const CPPHandleTypeInfo the_info {
            CPPTypeName(CPPTypeName::Simple, "char"),
            {}, {}, { CPPHandleTypeInfo::Pointer | CPPHandleTypeInfo::Const }
        };
        return &the_info;
    }
};

namespace HalideIR {

struct Expr;

/** Types in the halide type system. They can be ints, unsigned ints,
 * or floats of various bit-widths (the 'bits' field). They can also
 * be vectors of the same (by setting the 'lanes' field to something
 * larger than one). Front-end code shouldn't use vector
 * types. Instead vectorize a function. */
struct Type {
private:
    BaseType m_type;

public:
    /** Aliases for BaseTypeEnum values for legacy compatibility
     * and to match the Halide internal C++ style. */
    // @{
    static const BaseTypeEnum Int = base_type_int;
    static const BaseTypeEnum UInt = base_type_uint;
    static const BaseTypeEnum Float = base_type_float;
    static const BaseTypeEnum Handle = base_type_handle;
    // @}
    //
    /** Type to be printed when declaring handles of this type. */
    const CPPHandleTypeInfo* m_handle_type;

    /** The number of bytes required to store a single scalar value of this type. Ignores vector lanes. */
    int bytes() const { return (bits() + 7) / 8; }

    // Default ctor initializes everything to predictable-but-unlikely values
    Type()
        : m_type(Handle, 0, 0)
        , m_handle_type(nullptr)
    {
    }

    /** Construct a runtime representation of a Halide type from:
     * code: The fundamental type from an enum.
     * bits: The bit size of one element.
     * lanes: The number of vector elements in the type. */
    Type(BaseTypeEnum code, int bits, int lanes, const CPPHandleTypeInfo* handle_type = nullptr)
        : m_type(code, (uint8_t)bits, (uint16_t)lanes)
        , m_handle_type(handle_type)
    {
    }

    /** Trivial copy constructor. */
    Type(const Type& that) = default;

    /** Type is a wrapper around BaseType with more methods for use
     * inside the compiler. This simply constructs the wrapper around
     * the runtime value. */
    Type(const BaseType& that, const CPPHandleTypeInfo* handle_type = nullptr)
        : m_type(that)
        , m_handle_type(handle_type)
    {
    }

    /** Unwrap the runtime BaseType for use in runtime calls, etc.
     * Representation is exactly equivalent. */
    operator BaseType() const { return m_type; }

    /** Return the underlying data type of an element as an enum value. */
    BaseTypeEnum code() const { return (BaseTypeEnum)m_type.m_code; }

    /** Return the bit size of a single element of this type. */
    int bits() const { return m_type.m_bits; }

    /** Return the number of vector elements in this type. */
    int lanes() const { return m_type.m_lanes; }

    /** Return Type with same number of bits and lanes, but new_code for a type code. */
    Type with_code(BaseTypeEnum new_code) const
    {
        return Type(new_code, bits(), lanes(),
            (new_code == code()) ? m_handle_type : nullptr);
    }

    /** Return Type with same type code and lanes, but new_bits for the number of bits. */
    Type with_bits(int new_bits) const
    {
        return Type(code(), new_bits, lanes(),
            (new_bits == bits()) ? m_handle_type : nullptr);
    }

    /** Return Type with same type code and number of bits,
     * but new_lanes for the number of vector lanes. */
    Type with_lanes(int new_lanes) const
    {
        return Type(code(), bits(), new_lanes, m_handle_type);
    }

    /** Type to be printed when declaring handles of this type. */
    const CPPHandleTypeInfo* handle_type;

    /** Is this type boolean (represented as UInt(1))? */
    bool is_bool() const { return code() == UInt && bits() == 1; }

    /** Is this type a vector type? (lanes() != 1).
     * TODO(abadams): Decide what to do for lanes() == 0. */
    bool is_vector() const { return lanes() != 1; }

    /** Is this type a scalar type? (lanes() == 1).
     * TODO(abadams): Decide what to do for lanes() == 0. */
    bool is_scalar() const { return lanes() == 1; }

    /** Is this type a floating point type (float or double). */
    bool is_float() const { return code() == Float; }

    /** Is this type a signed integer type? */
    bool is_int() const { return code() == Int; }

    /** Is this type an unsigned integer type? */
    bool is_uint() const { return code() == UInt; }

    /** Is this type an opaque handle type (void *) */
    bool is_handle() const { return code() == Handle; }

    /** Check that the type name of two handles matches. */
    EXPORT bool same_handle_type(const Type& other) const;

    /** Compare two types for equality */
    bool operator==(const Type& other) const
    {
        return code() == other.code() && bits() == other.bits() && lanes() == other.lanes() && (code() != Handle || same_handle_type(other));
    }

    /** Compare two types for inequality */
    bool operator!=(const Type& other) const
    {
        return code() != other.code() || bits() != other.bits() || lanes() != other.lanes() || (code() == Handle && !same_handle_type(other));
    }

    /** Produce the scalar type (that of a single element) of this vector type */
    Type element_of() const
    {
        return with_lanes(1);
    }

    /** Can this type represent all values of another type? */
    EXPORT bool can_represent(Type other) const;

    /** Can this type represent a particular constant? */
    // @{
    EXPORT bool can_represent(double x) const;
    EXPORT bool can_represent(int64_t x) const;
    EXPORT bool can_represent(uint64_t x) const;
    // @}

    /** Check if an integer constant value is the maximum or minimum
     * representable value for this type. */
    // @{
    EXPORT bool is_max(uint64_t) const;
    EXPORT bool is_max(int64_t) const;
    EXPORT bool is_min(uint64_t) const;
    EXPORT bool is_min(int64_t) const;
    // @}

    /** Return an expression which is the maximum value of this type */
    EXPORT Expr max() const;

    /** Return an expression which is the minimum value of this type */
    EXPORT Expr min() const;
};

/** Constructing a signed integer type */
inline Type Int(int bits, int lanes = 1)
{
    return Type(Type::Int, bits, lanes);
}

/** Constructing an unsigned integer type */
inline Type UInt(int bits, int lanes = 1)
{
    return Type(Type::UInt, bits, lanes);
}

/** Construct a floating-point type */
inline Type Float(int bits, int lanes = 1)
{
    return Type(Type::Float, bits, lanes);
}

/** Construct a boolean type */
inline Type Bool(int lanes = 1)
{
    return UInt(1, lanes);
}

/** Construct a handle type */
inline Type Handle(int lanes = 1, const CPPHandleTypeInfo* handle_type = nullptr)
{
    return Type(Type::Handle, 64, lanes, handle_type);
}

/** Construct the halide equivalent of a C type */
template <typename T>
inline Type type_of()
{
    return Type(BaseTypeCast<T>(), CPPHandleTraits<T>::type_info());
}

} // namespace HalideIR
