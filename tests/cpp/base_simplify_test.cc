#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdio.h>

#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include "halide/arithmetic/Simplify.h"
#include "halide/ir/Expr.h"
#include "halide/ir/IR.h"
#include "halide/ir/IREquality.h"
#include "halide/ir/IROperator.h"

using std::map;
using std::ostringstream;
using std::pair;
using std::string;
using std::vector;

using namespace HalideIR;
using namespace HalideIR::Internal;

template <typename T>
class TD;

void check(const Expr& a, const Expr& b)
{
    //debug(0) << "Checking that " << a << " -> " << b << "\n";
    Expr simpler = simplify(a);
    if (!equal(simpler, b)) {
        internal_error
            << "\nSimplification failure:\n"
            << "Input: " << a << '\n'
            << "Output: " << simpler << '\n'
            << "Expected output: " << b << '\n';
    }
}

void check(const Stmt& a, const Stmt& b)
{
    //debug(0) << "Checking that " << a << " -> " << b << "\n";
    Stmt simpler = simplify(a);
    if (!equal(simpler, b)) {
        internal_error
            << "\nSimplification failure:\n"
            << "Input: " << a << '\n'
            << "Output: " << simpler << '\n'
            << "Expected output: " << b << '\n';
    }
}

void check_in_bounds(const Expr& a, const Expr& b, const Scope<Interval>& bi)
{
    //debug(0) << "Checking that " << a << " -> " << b << "\n";
    Expr simpler = simplify(a, true, bi);
    if (!equal(simpler, b)) {
        internal_error
            << "\nSimplification failure:\n"
            << "Input: " << a << '\n'
            << "Output: " << simpler << '\n'
            << "Expected output: " << b << '\n';
    }
}

// Helper functions to use in the tests below
Expr interleave_vectors(const vector<Expr>& e)
{
    return Shuffle::make_interleave(e);
}

Expr concat_vectors(const vector<Expr>& e)
{
    return Shuffle::make_concat(e);
}

Expr slice(const Expr& e, int begin, int stride, int w)
{
    return Shuffle::make_slice(e, begin, stride, w);
}

Expr ramp(const Expr& base, const Expr& stride, int w)
{
    return Ramp::make(base, stride, w);
}

Expr broadcast(const Expr& base, int w)
{
    return Broadcast::make(base, w);
}

VarExpr Var(string name)
{
    return Variable::make(Int(32), name);
}

void check_casts()
{
    Expr x = Var("x");

    check(cast(Int(32), cast(Int(32), x)), x);
    check(cast(Float(32), 3), 3.0f);
    check(cast(Int(32), 5.0f), 5);

    check(cast(Int(32), cast(Int(8), 3)), 3);
    check(cast(Int(32), cast(Int(8), 1232)), -48);

    // Check redundant casts
    check(cast(Float(32), cast(Float(64), x)), cast(Float(32), x));
    check(cast(Int(16), cast(Int(32), x)), cast(Int(16), x));
    check(cast(Int(16), cast(UInt(32), x)), cast(Int(16), x));
    check(cast(UInt(16), cast(Int(32), x)), cast(UInt(16), x));
    check(cast(UInt(16), cast(UInt(32), x)), cast(UInt(16), x));

    // Check evaluation of constant expressions involving casts
    check(cast(UInt(16), 53) + cast(UInt(16), 87), make_const(UInt(16), 140));
    check(cast(Int(8), 127) + cast(Int(8), 1), make_const(Int(8), -128));
    check(cast(UInt(16), -1) - cast(UInt(16), 1), make_const(UInt(16), 65534));
    check(cast(Int(16), 4) * cast(Int(16), -5), make_const(Int(16), -20));
    check(cast(Int(16), 16) / cast(Int(16), 4), make_const(Int(16), 4));
    check(cast(Int(16), 23) % cast(Int(16), 5), make_const(Int(16), 3));
    check(min(cast(Int(16), 30000), cast(Int(16), -123)), make_const(Int(16), -123));
    check(max(cast(Int(16), 30000), cast(Int(16), 65000)), make_const(Int(16), 30000));
    check(cast(UInt(16), -1) == cast(UInt(16), 65535), const_true());
    check(cast(UInt(16), 65) == cast(UInt(16), 66), const_false());
    check(cast(UInt(16), -1) < cast(UInt(16), 65535), const_false());
    check(cast(UInt(16), 65) < cast(UInt(16), 66), const_true());
    check(cast(UInt(16), 123.4f), make_const(UInt(16), 123));
    check(cast(Float(32), cast(UInt(16), 123456.0f)), 57920.0f);
    // Specific checks for 32 bit unsigned expressions - ensure simplifications are actually unsigned.
    // 4000000000 (4 billion) is less than 2^32 but more than 2^31.  As an int, it is negative.
    check(cast(UInt(32), (int)4000000000UL) + cast(UInt(32), 5), make_const(UInt(32), (int)4000000005UL));
    check(cast(UInt(32), (int)4000000000UL) - cast(UInt(32), 5), make_const(UInt(32), (int)3999999995UL));
    check(cast(UInt(32), (int)4000000000UL) / cast(UInt(32), 5), make_const(UInt(32), 800000000));
    check(cast(UInt(32), 800000000) * cast(UInt(32), 5), make_const(UInt(32), (int)4000000000UL));
    check(cast(UInt(32), (int)4000000023UL) % cast(UInt(32), 100), make_const(UInt(32), 23));
    check(min(cast(UInt(32), (int)4000000023UL), cast(UInt(32), 1000)), make_const(UInt(32), (int)1000));
    check(max(cast(UInt(32), (int)4000000023UL), cast(UInt(32), 1000)), make_const(UInt(32), (int)4000000023UL));
    check(cast(UInt(32), (int)4000000023UL) < cast(UInt(32), 1000), const_false());
    check(cast(UInt(32), (int)4000000023UL) == cast(UInt(32), 1000), const_false());

    check(cast(Float(64), 0.5f), Expr(0.5));
    check((x - cast(Float(64), 0.5f)) * (x - cast(Float(64), 0.5f)),
        (x + Expr(-0.5)) * (x + Expr(-0.5)));

    check(cast(Int(64, 3), ramp(5.5f, 2.0f, 3)),
        cast(Int(64, 3), ramp(5.5f, 2.0f, 3)));
    check(cast(Int(64, 3), ramp(x, 2, 3)),
        ramp(cast(Int(64), x), cast(Int(64), 2), 3));

    // Check cancellations can occur through casts
    check(cast(Int(64), x + 1) - cast(Int(64), x), cast(Int(64), 1));
    check(cast(Int(64), 1 + x) - cast(Int(64), x), cast(Int(64), 1));
    // But only when overflow is undefined for the type
    check(cast(UInt(8), x + 1) - cast(UInt(8), x),
        cast(UInt(8), x + 1) - cast(UInt(8), x));
}

void check_algebra()
{
    Expr x = Var("x"), y = Var("y"), z = Var("z"), w = Var("w"), v = Var("v");
    Expr xf = cast<float>(x);
    Expr yf = cast<float>(y);
    Expr t = const_true(), f = const_false();

    check(3 + x, x + 3);
    check(x + 0, x);
    check(0 + x, x);
    check(Expr(ramp(x, 2, 3)) + Expr(ramp(y, 4, 3)), ramp(x + y, 6, 3));
    check(Expr(broadcast(4.0f, 5)) + Expr(ramp(3.25f, 4.5f, 5)), ramp(7.25f, 4.5f, 5));
    check(Expr(ramp(3.25f, 4.5f, 5)) + Expr(broadcast(4.0f, 5)), ramp(7.25f, 4.5f, 5));
    check(Expr(broadcast(3, 3)) + Expr(broadcast(1, 3)), broadcast(4, 3));
    check((x + 3) + 4, x + 7);
    check(4 + (3 + x), x + 7);
    check((x + 3) + y, (x + y) + 3);
    check(y + (x + 3), (y + x) + 3);
    check((3 - x) + x, 3);
    check(x + (3 - x), 3);
    check(x * y + x * z, x * (y + z));
    check(x * y + z * x, x * (y + z));
    check(y * x + x * z, x * (y + z));
    check(y * x + z * x, x * (y + z));

    check(x - 0, x);
    check((x / y) - (x / y), 0);
    check(x - 2, x + (-2));
    check(Expr(ramp(x, 2, 3)) - Expr(ramp(y, 4, 3)), ramp(x - y, -2, 3));
    check(Expr(broadcast(4.0f, 5)) - Expr(ramp(3.25f, 4.5f, 5)), ramp(0.75f, -4.5f, 5));
    check(Expr(ramp(3.25f, 4.5f, 5)) - Expr(broadcast(4.0f, 5)), ramp(-0.75f, 4.5f, 5));
    check(Expr(broadcast(3, 3)) - Expr(broadcast(1, 3)), broadcast(2, 3));
    check((x + y) - x, y);
    check((x + y) - y, x);
    check(x - (x + y), 0 - y);
    check(x - (y + x), 0 - y);
    check((x + 3) - 2, x + 1);
    check((x + 3) - y, (x - y) + 3);
    check((x - 3) - y, (x - y) + (-3));
    check(x - (y - 2), (x - y) + 2);
    check(3 - (y - 2), 5 - y);
    check(x - (0 - y), x + y);
    check(x + (0 - y), x - y);
    check((0 - x) + y, y - x);
    check(x * y - x * z, x * (y - z));
    check(x * y - z * x, x * (y - z));
    check(y * x - x * z, x * (y - z));
    check(y * x - z * x, x * (y - z));
    check(x - y * -2, x + y * 2);
    check(x + y * -2, x - y * 2);
    check(x * -2 + y, y - x * 2);
    check(xf - yf * -2.0f, xf + y * 2.0f);
    check(xf + yf * -2.0f, xf - y * 2.0f);
    check(xf * -2.0f + yf, yf - x * 2.0f);

    check(x - (x / 8) * 8, x % 8);
    check((x / 8) * 8 - x, -(x % 8));
    check((x / 8) * 8 < x + y, 0 < x % 8 + y);
    check((x / 8) * 8 < x - y, y < x % 8);
    check((x / 8) * 8 < x, 0 < x % 8);
    check(((x + 3) / 8) * 8 < x + y, 3 < (x + 3) % 8 + y);
    check(((x + 3) / 8) * 8 < x - y, y < (x + 3) % 8 + (-3));
    check(((x + 3) / 8) * 8 < x, 3 < (x + 3) % 8);

    check(x * 0, 0);
    check(0 * x, 0);
    check(x * 1, x);
    check(1 * x, x);
    check(Expr(2.0f) * 4.0f, 8.0f);
    check(Expr(2) * 4, 8);
    check((3 * x) * 4, x * 12);
    check(4 * (3 + x), x * 4 + 12);
    check(Expr(broadcast(4.0f, 5)) * Expr(ramp(3.0f, 4.0f, 5)), ramp(12.0f, 16.0f, 5));
    check(Expr(ramp(3.0f, 4.0f, 5)) * Expr(broadcast(2.0f, 5)), ramp(6.0f, 8.0f, 5));
    check(Expr(broadcast(3, 3)) * Expr(broadcast(2, 3)), broadcast(6, 3));

    check(x * y + x, x * (y + 1));
    check(x * y - x, x * (y + -1));
    check(x + x * y, x * (y + 1));
    check(x - x * y, x * (1 - y));
    check(x * y + y, (x + 1) * y);
    check(x * y - y, (x + -1) * y);
    check(y + x * y, (x + 1) * y);
    check(y - x * y, (1 - x) * y);

    check(0 / x, 0);
    check(x / 1, x);
    check(x / x, 1);
    check((-1) / x, select(x < 0, 1, -1));
    check(Expr(7) / 3, 2);
    check(Expr(6.0f) / 2.0f, 3.0f);
    check((x / 3) / 4, x / 12);
    check((x * 4) / 2, x * 2);
    check((x * 2) / 4, x / 2);
    check((x * 4 + y) / 2, x * 2 + y / 2);
    check((y + x * 4) / 2, y / 2 + x * 2);
    check((x * 4 - y) / 2, x * 2 + (0 - y) / 2);
    check((y - x * 4) / 2, y / 2 - x * 2);
    check((x + 3) / 2 + 7, (x + 17) / 2);
    check((x / 2 + 3) / 5, (x + 6) / 10);
    check((x + 8) / 2, x / 2 + 4);
    check((x - y) * -2, (y - x) * 2);
    check((xf - yf) * -2.0f, (yf - xf) * 2.0f);

    // Pull terms that are a multiple of the divisor out of a ternary expression
    check(((x * 4 + y) + z) / 2, x * 2 + (y + z) / 2);
    check(((x * 4 - y) + z) / 2, x * 2 + (z - y) / 2);
    check(((x * 4 + y) - z) / 2, x * 2 + (y - z) / 2);
    check(((x * 4 - y) - z) / 2, x * 2 + (0 - y - z) / 2);
    check((x + (y * 4 + z)) / 2, y * 2 + (x + z) / 2);
    check((x + (y * 4 - z)) / 2, y * 2 + (x - z) / 2);
    check((x - (y * 4 + z)) / 2, (x - z) / 2 - y * 2);
    check((x - (y * 4 - z)) / 2, (x + z) / 2 - y * 2);

    // Cancellations in non-const integer divisions
    check((x * y) / x, y);
    check((y * x) / x, y);
    check((x * y + z) / x, y + z / x);
    check((y * x + z) / x, y + z / x);
    check((z + x * y) / x, z / x + y);
    check((z + y * x) / x, z / x + y);
    check((x * y - z) / x, y + (-z) / x);
    check((y * x - z) / x, y + (-z) / x);
    check((z - x * y) / x, z / x - y);
    check((z - y * x) / x, z / x - y);

    check((x + y) / x, y / x + 1);
    check((y + x) / x, y / x + 1);
    check((x - y) / x, (-y) / x + 1);
    check((y - x) / x, y / x + (-1));

    check(((x + y) + z) / x, (y + z) / x + 1);
    check(((y + x) + z) / x, (y + z) / x + 1);
    check((y + (x + z)) / x, (y + z) / x + 1);
    check((y + (z + x)) / x, (y + z) / x + 1);

    check(xf / 4.0f, xf * 0.25f);

    // Some quaternary rules with cancellations
    check((x + y) - (z + y), x - z);
    check((x + y) - (y + z), x - z);
    check((y + x) - (z + y), x - z);
    check((y + x) - (y + z), x - z);

    check((x - y) - (z - y), x - z);
    check((y - z) - (y - x), x - z);

    check((x * 8) % 4, 0);
    check((x * 8 + y) % 4, y % 4);
    // check((y + 8) % 4, y % 4);
    check((y + x * 8) % 4, y % 4);
    check((y * 16 + 13) % 2, 1);

    // Check an optimization important for fusing dimensions
    check((x / 3) * 3 + x % 3, x);
    check(x % 3 + (x / 3) * 3, x);

    check(((x / 3) * 3 + y) + x % 3, x + y);
    check((x % 3 + y) + (x / 3) * 3, x + y);

    check((y + x % 3) + (x / 3) * 3, y + x);
    check((y + (x / 3 * 3)) + x % 3, y + x);

    // Almost-cancellations through integer divisions. These rules all
    // deduplicate x and wrap it in a modulo operator, neutering it
    // for the purposes of bounds inference. Patterns below look
    // confusing, but were brute-force tested.
    check((x + 17) / 3 - (x + 7) / 3, ((x + 1) % 3 + 10) / 3);
    check((x + 17) / 3 - (x + y) / 3, (19 - y - (x + 2) % 3) / 3);
    check((x + y) / 3 - (x + 7) / 3, ((x + 1) % 3 + y + -7) / 3);
    check(x / 3 - (x + y) / 3, (2 - y - x % 3) / 3);
    check((x + y) / 3 - x / 3, (x % 3 + y) / 3);
    check(x / 3 - (x + 7) / 3, (-5 - x % 3) / 3);
    check((x + 17) / 3 - x / 3, (x % 3 + 17) / 3);
    check((x + 17) / 3 - (x - y) / 3, (y - (x + 2) % 3 + 19) / 3);
    check((x - y) / 3 - (x + 7) / 3, ((x + 1) % 3 - y + (-7)) / 3);
    check(x / 3 - (x - y) / 3, (y - x % 3 + 2) / 3);
    check((x - y) / 3 - x / 3, (x % 3 - y) / 3);

    // Check some specific expressions involving div and mod
    check(Expr(23) / 4, Expr(5));
    check(Expr(-23) / 4, Expr(-6));
    check(Expr(-23) / -4, Expr(6));
    check(Expr(23) / -4, Expr(-5));
    check(Expr(-2000000000) / 1000000001, Expr(-2));
    check(Expr(23) % 4, Expr(3));
    check(Expr(-23) % 4, Expr(1));
    check(Expr(-23) % -4, Expr(1));
    check(Expr(23) % -4, Expr(3));
    check(Expr(-2000000000) % 1000000001, Expr(2));

    check(Expr(3) + Expr(8), 11);
    check(Expr(3.25f) + Expr(7.75f), 11.0f);

    check(Expr(7) % 2, 1);
    check(Expr(7.25f) % 2.0f, 1.25f);
    check(Expr(-7.25f) % 2.0f, 0.75f);
    check(Expr(-7.25f) % -2.0f, -1.25f);
    check(Expr(7.25f) % -2.0f, -0.75f);
}

void check_vectors()
{
    Expr x = Var("x"), y = Var("y"), z = Var("z");

    check(Expr(broadcast(y, 4)) / Expr(broadcast(x, 4)),
        Expr(broadcast(y / x, 4)));
    check(Expr(ramp(x, 4, 4)) / 2, ramp(x / 2, 2, 4));
    check(Expr(ramp(x, -4, 7)) / 2, ramp(x / 2, -2, 7));
    check(Expr(ramp(x, 4, 5)) / -2, ramp(x / -2, -2, 5));
    check(Expr(ramp(x, -8, 5)) / -2, ramp(x / -2, 4, 5));

    check(Expr(ramp(4 * x, 1, 4)) / 4, broadcast(x, 4));
    check(Expr(ramp(x * 4, 1, 3)) / 4, broadcast(x, 3));
    check(Expr(ramp(x * 8, 2, 4)) / 8, broadcast(x, 4));
    check(Expr(ramp(x * 8, 3, 3)) / 8, broadcast(x, 3));
    check(Expr(ramp(0, 1, 8)) % 16, Expr(ramp(0, 1, 8)));
    check(Expr(ramp(8, 1, 8)) % 16, Expr(ramp(8, 1, 8)));
    check(Expr(ramp(9, 1, 8)) % 16, Expr(ramp(9, 1, 8)) % 16);
    check(Expr(ramp(16, 1, 8)) % 16, Expr(ramp(0, 1, 8)));
    check(Expr(ramp(0, 1, 8)) % 8, Expr(ramp(0, 1, 8)));
    check(Expr(ramp(x * 8 + 17, 1, 4)) % 8, Expr(ramp(1, 1, 4)));
    check(Expr(ramp(x * 8 + 17, 1, 8)) % 8, Expr(ramp(1, 1, 8) % 8));

    check(Expr(broadcast(x, 4)) % Expr(broadcast(y, 4)),
        Expr(broadcast(x % y, 4)));
    check(Expr(ramp(x, 2, 4)) % (broadcast(2, 4)),
        broadcast(x % 2, 4));
    check(Expr(ramp(2 * x + 1, 4, 4)) % (broadcast(2, 4)),
        broadcast(1, 4));

    check(ramp(0, 1, 4) == broadcast(2, 4),
        ramp(-2, 1, 4) == broadcast(0, 4));

    {
        Expr test = select(ramp(const_true(), const_true(), 2),
                        ramp(const_false(), const_true(), 2),
                        broadcast(const_false(), 2))
            == broadcast(const_false(), 2);
        Expr expected = !(ramp(const_true(), const_true(), 2)) || (ramp(const_false(), const_true(), 2) == broadcast(const_false(), 2));
        check(test, expected);
    }

    {
        Expr test = select(ramp(const_true(), const_true(), 2),
                        broadcast(const_true(), 2),
                        ramp(const_false(), const_true(), 2))
            == broadcast(const_false(), 2);
        Expr expected = (!ramp(const_true(), const_true(), 2)) && (ramp(const_false(), const_true(), 2) == broadcast(const_false(), 2));
        check(test, expected);
    }
}

void check_bounds()
{
    Expr x = Var("x"), y = Var("y"), z = Var("z");

    check(min(Expr(7), 3), 3);
    check(min(Expr(4.25f), 1.25f), 1.25f);
    check(min(broadcast(x, 4), broadcast(y, 4)),
        broadcast(min(x, y), 4));
    check(min(x, x + 3), x);
    check(min(x + 4, x), x);
    check(min(x - 1, x + 2), x + (-1));
    check(min(7, min(x, 3)), min(x, 3));
    check(min(min(x, y), x), min(x, y));
    check(min(min(x, y), y), min(x, y));
    check(min(x, min(x, y)), min(x, y));
    check(min(y, min(x, y)), min(x, y));

    check(max(Expr(7), 3), 7);
    check(max(Expr(4.25f), 1.25f), 4.25f);
    check(max(broadcast(x, 4), broadcast(y, 4)),
        broadcast(max(x, y), 4));
    check(max(x, x + 3), x + 3);
    check(max(x + 4, x), x + 4);
    check(max(x - 1, x + 2), x + 2);
    check(max(7, max(x, 3)), max(x, 7));
    check(max(max(x, y), x), max(x, y));
    check(max(max(x, y), y), max(x, y));
    check(max(x, max(x, y)), max(x, y));
    check(max(y, max(x, y)), max(x, y));

    // Check that simplifier can recognise instances where the extremes of the
    // datatype appear as constants in comparisons, Min and Max expressions.
    // The result of min/max with extreme is known to be either the extreme or
    // the other expression.  The result of < or > comparison is known to be true or false.
    check(x <= Int(32).max(), const_true());
    check(cast(Int(16), x) >= Int(16).min(), const_true());
    check(x < Int(32).min(), const_false());
    check(min(cast(UInt(16), x), cast(UInt(16), 65535)), cast(UInt(16), x));
    check(min(x, Int(32).max()), x);
    check(min(Int(32).min(), x), Int(32).min());
    check(max(cast(Int(8), x), cast(Int(8), -128)), cast(Int(8), x));
    check(max(x, Int(32).min()), x);
    check(max(x, Int(32).max()), Int(32).max());
    // Check that non-extremes do not lead to incorrect simplification
    check(max(cast(Int(8), x), cast(Int(8), -127)), max(cast(Int(8), x), make_const(Int(8), -127)));

    // Some quaternary rules with cancellations
    check((x + y) - (z + y), x - z);
    check((x + y) - (y + z), x - z);
    check((y + x) - (z + y), x - z);
    check((y + x) - (y + z), x - z);

    check((x - y) - (z - y), x - z);
    check((y - z) - (y - x), x - z);

    check((x + 3) / 4 - (x + 2) / 4, ((x + 2) % 4 + 1) / 4);

    check(x - min(x + y, z), max(-y, x - z));
    check(x - min(y + x, z), max(-y, x - z));
    check(x - min(z, x + y), max(-y, x - z));
    check(x - min(z, y + x), max(-y, x - z));

    check(min(x + y, z) - x, min(y, z - x));
    check(min(y + x, z) - x, min(y, z - x));
    check(min(z, x + y) - x, min(y, z - x));
    check(min(z, y + x) - x, min(y, z - x));

    check(min(x + y, z + y), min(x, z) + y);
    check(min(y + x, z + y), min(x, z) + y);
    check(min(x + y, y + z), min(x, z) + y);
    check(min(y + x, y + z), min(x, z) + y);

    check(min(x, y) - min(y, x), 0);
    check(max(x, y) - max(y, x), 0);

    check(min(123 - x, 1 - x), 1 - x);
    check(max(123 - x, 1 - x), 123 - x);

    check(min(x * 43, y * 43), min(x, y) * 43);
    check(max(x * 43, y * 43), max(x, y) * 43);
    check(min(x * -43, y * -43), max(x, y) * -43);
    check(max(x * -43, y * -43), min(x, y) * -43);

    check(min(min(x, 4), y), min(min(x, y), 4));
    check(max(max(x, 4), y), max(max(x, y), 4));

    check(min(x * 8, 24), min(x, 3) * 8);
    check(max(x * 8, 24), max(x, 3) * 8);
    check(min(x * -8, 24), max(x, -3) * -8);
    check(max(x * -8, 24), min(x, -3) * -8);

    check(min(clamp(x, -10, 14), clamp(y, -10, 14)), clamp(min(x, y), -10, 14));

    check(min(x / 4, y / 4), min(x, y) / 4);
    check(max(x / 4, y / 4), max(x, y) / 4);

    check(min(x / (-4), y / (-4)), max(x, y) / (-4));
    check(max(x / (-4), y / (-4)), min(x, y) / (-4));

    // Min and max of clamped expressions
    check(min(clamp(x + 1, y, z), clamp(x - 1, y, z)), clamp(x + (-1), y, z));
    check(max(clamp(x + 1, y, z), clamp(x - 1, y, z)), clamp(x + 1, y, z));

    // Additions that cancel a term inside a min or max
    check(x + min(y - x, z), min(y, z + x));
    check(x + max(y - x, z), max(y, z + x));
    check(min(y + (-2), z) + 2, min(y, z + 2));
    check(max(y + (-2), z) + 2, max(y, z + 2));

    check(x + min(y - x, z), min(y, z + x));
    check(x + max(y - x, z), max(y, z + x));
    check(min(y + (-2), z) + 2, min(y, z + 2));
    check(max(y + (-2), z) + 2, max(y, z + 2));

    // Min/Max distributive law
    check(max(max(x, y), max(x, z)), max(max(y, z), x));
    check(min(max(x, y), max(x, z)), max(min(y, z), x));
    check(min(min(x, y), min(x, z)), min(min(y, z), x));
    check(max(min(x, y), min(x, z)), min(max(y, z), x));

    // Mins of expressions and rounded up versions of them
    check(min(((x + 7) / 8) * 8, x), x);
    check(min(x, ((x + 7) / 8) * 8), x);

    check(min(((x + 7) / 8) * 8, max(x, 8)), max(x, 8));
    check(min(max(x, 8), ((x + 7) / 8) * 8), max(x, 8));

    check(min(x, likely(x)), likely(x));
    check(min(likely(x), x), likely(x));
    check(max(x, likely(x)), likely(x));
    check(max(likely(x), x), likely(x));
    check(select(x > y, likely(x), x), likely(x));
    check(select(x > y, x, likely(x)), likely(x));

    check(min(x + 1, y) - min(x, y - 1), 1);
    check(max(x + 1, y) - max(x, y - 1), 1);
    check(min(x + 1, y) - min(y - 1, x), 1);
    check(max(x + 1, y) - max(y - 1, x), 1);

    // min and max on constant ramp v broadcast
    check(max(ramp(0, 1, 8), 0), ramp(0, 1, 8));
    check(min(ramp(0, 1, 8), 7), ramp(0, 1, 8));
    check(max(ramp(0, 1, 8), 7), broadcast(7, 8));
    check(min(ramp(0, 1, 8), 0), broadcast(0, 8));
    check(min(ramp(0, 1, 8), 4), min(ramp(0, 1, 8), 4));

    check(max(ramp(7, -1, 8), 0), ramp(7, -1, 8));
    check(min(ramp(7, -1, 8), 7), ramp(7, -1, 8));
    check(max(ramp(7, -1, 8), 7), broadcast(7, 8));
    check(min(ramp(7, -1, 8), 0), broadcast(0, 8));
    check(min(ramp(7, -1, 8), 4), min(ramp(7, -1, 8), 4));

    check(max(0, ramp(0, 1, 8)), ramp(0, 1, 8));
    check(min(7, ramp(0, 1, 8)), ramp(0, 1, 8));

    check(min(8 - x, 2), 8 - max(x, 6));
    check(max(3, 77 - x), 77 - min(x, 74));
    check(min(max(8 - x, 0), 8), 8 - max(min(x, 8), 0));

    check(x - min(x, 2), max(x + -2, 0));
    check(x - max(x, 2), min(x + -2, 0));
    check(min(x, 2) - x, 2 - max(x, 2));
    check(max(x, 2) - x, 2 - min(x, 2));
    check(x - min(2, x), max(x + -2, 0));
    check(x - max(2, x), min(x + -2, 0));
    check(min(2, x) - x, 2 - max(x, 2));
    check(max(2, x) - x, 2 - min(x, 2));

    check(max(min(x, y), x), x);
    check(max(min(x, y), y), y);
    check(min(max(x, y), x), x);
    check(min(max(x, y), y), y);
    check(max(min(x, y), x) + y, x + y);

    check(max(min(max(x, y), z), y), max(min(x, z), y));
    check(max(min(z, max(x, y)), y), max(min(x, z), y));
    check(max(y, min(max(x, y), z)), max(min(x, z), y));
    check(max(y, min(z, max(x, y))), max(min(x, z), y));

    check(max(min(max(y, x), z), y), max(min(x, z), y));
    check(max(min(z, max(y, x)), y), max(min(x, z), y));
    check(max(y, min(max(y, x), z)), max(min(x, z), y));
    check(max(y, min(z, max(y, x))), max(min(x, z), y));

    check(min(max(min(x, y), z), y), min(max(x, z), y));
    check(min(max(z, min(x, y)), y), min(max(x, z), y));
    check(min(y, max(min(x, y), z)), min(max(x, z), y));
    check(min(y, max(z, min(x, y))), min(max(x, z), y));

    check(min(max(min(y, x), z), y), min(max(x, z), y));
    check(min(max(z, min(y, x)), y), min(max(x, z), y));
    check(min(y, max(min(y, x), z)), min(max(x, z), y));
    check(min(y, max(z, min(y, x))), min(max(x, z), y));

    {
        Expr one = broadcast(cast(Int(16), 1), 64);
        Expr three = broadcast(cast(Int(16), 3), 64);
        Expr four = broadcast(cast(Int(16), 4), 64);
        Expr five = broadcast(cast(Int(16), 5), 64);
        Expr v1 = Variable::make(Int(16).with_lanes(64), "x");
        Expr v2 = Variable::make(Int(16).with_lanes(64), "y");

        // Bound: [-4, 4]
        std::vector<Expr> clamped = {
            max(min(v1, four), -four),
            max(-four, min(v1, four)),
            min(max(v1, -four), four),
            min(four, max(v1, -four)),
            clamp(v1, -four, four)
        };

        for (size_t i = 0; i < clamped.size(); ++i) {
            // min(v, 4) where v=[-4, 4] -> v
            check(min(clamped[i], four), simplify(clamped[i]));
            // min(v, 5) where v=[-4, 4] -> v
            check(min(clamped[i], five), simplify(clamped[i]));
            // min(v, 3) where v=[-4, 4] -> min(v, 3)
            check(min(clamped[i], three), simplify(min(clamped[i], three)));
            // min(v, -5) where v=[-4, 4] -> -5
            check(min(clamped[i], -five), simplify(-five));
        }

        for (size_t i = 0; i < clamped.size(); ++i) {
            // max(v, 4) where v=[-4, 4] -> 4
            check(max(clamped[i], four), simplify(four));
            // max(v, 5) where v=[-4, 4] -> 5
            check(max(clamped[i], five), simplify(five));
            // max(v, 3) where v=[-4, 4] -> max(v, 3)
            check(max(clamped[i], three), simplify(max(clamped[i], three)));
            // max(v, -5) where v=[-4, 4] -> v
            check(max(clamped[i], -five), simplify(clamped[i]));
        }

        for (size_t i = 0; i < clamped.size(); ++i) {
            // max(min(v, 5), -5) where v=[-4, 4] -> v
            check(max(min(clamped[i], five), -five), simplify(clamped[i]));
            // max(min(v, 5), 5) where v=[-4, 4] -> 5
            check(max(min(clamped[i], five), five), simplify(five));

            // max(min(v, -5), -5) where v=[-4, 4] -> -5
            check(max(min(clamped[i], -five), -five), simplify(-five));
            // max(min(v, -5), 5) where v=[-4, 4] -> 5
            check(max(min(clamped[i], -five), five), simplify(five));
            // max(min(v, -5), 3) where v=[-4, 4] -> 3
            check(max(min(clamped[2], -five), three), simplify(three));
        }

        // max(min(v, 5), 3) where v=[-4, 4] -> max(v, 3)
        check(max(min(clamped[2], five), three), simplify(max(clamped[2], three)));

        // max(min(v, 5), 3) where v=[-4, 4] -> max(v, 3) -> v=[3, 4]
        // There is simplification rule that will simplify max(max(min(x, 4), -4), 3)
        // further into max(min(x, 4), 3)
        check(max(min(clamped[0], five), three), simplify(max(min(v1, four), three)));

        for (size_t i = 0; i < clamped.size(); ++i) {
            // min(v + 1, 4) where v=[-4, 4] -> min(v + 1, 4)
            check(min(clamped[i] + one, four), simplify(min(clamped[i] + one, four)));
            // min(v + 1, 5) where v=[-4, 4] -> v + 1
            check(min(clamped[i] + one, five), simplify(clamped[i] + one));
            // min(v + 1, -4) where v=[-4, 4] -> -4
            check(min(clamped[i] + one, -four), simplify(-four));
            // max(min(v + 1, 4), -4) where v=[-4, 4] -> min(v + 1, 4)
            check(max(min(clamped[i] + one, four), -four), simplify(min(clamped[i] + one, four)));
        }
        for (size_t i = 0; i < clamped.size(); ++i) {
            // max(v + 1, 4) where v=[-4, 4] -> max(v + 1, 4)
            check(max(clamped[i] + one, four), simplify(max(clamped[i] + one, four)));
            // max(v + 1, 5) where v=[-4, 4] -> 5
            check(max(clamped[i] + one, five), simplify(five));
            // max(v + 1, -4) where v=[-4, 4] -> -v + 1
            check(max(clamped[i] + one, -four), simplify(clamped[i] + one));
            // min(max(v + 1, -4), 4) where v=[-4, 4] -> min(v + 1, 4)
            check(min(max(clamped[i] + one, -four), four), simplify(min(clamped[i] + one, four)));
        }

        Expr t1 = clamp(v1, one, four);
        Expr t2 = clamp(v1, -five, -four);
        check(min(max(min(v2, t1), t2), five), simplify(max(min(t1, v2), t2)));
    }

    {
        Expr xv = Variable::make(Int(16).with_lanes(64), "x");
        Expr yv = Variable::make(Int(16).with_lanes(64), "y");
        Expr zv = Variable::make(Int(16).with_lanes(64), "z");

        // min(min(x, broadcast(y, n)), broadcast(z, n))) -> min(x, broadcast(min(y, z), n))
        check(min(min(xv, broadcast(y, 64)), broadcast(z, 64)), min(xv, broadcast(min(y, z), 64)));
        // min(min(broadcast(x, n), y), broadcast(z, n))) -> min(y, broadcast(min(x, z), n))
        check(min(min(broadcast(x, 64), yv), broadcast(z, 64)), min(yv, broadcast(min(x, z), 64)));
        // min(broadcast(x, n), min(y, broadcast(z, n)))) -> min(y, broadcast(min(x, z), n))
        check(min(broadcast(x, 64), min(yv, broadcast(z, 64))), min(yv, broadcast(min(z, x), 64)));
        // min(broadcast(x, n), min(broadcast(y, n), z))) -> min(z, broadcast(min(x, y), n))
        check(min(broadcast(x, 64), min(broadcast(y, 64), zv)), min(zv, broadcast(min(y, x), 64)));

        // max(max(x, broadcast(y, n)), broadcast(z, n))) -> max(x, broadcast(max(y, z), n))
        check(max(max(xv, broadcast(y, 64)), broadcast(z, 64)), max(xv, broadcast(max(y, z), 64)));
        // max(max(broadcast(x, n), y), broadcast(z, n))) -> max(y, broadcast(max(x, z), n))
        check(max(max(broadcast(x, 64), yv), broadcast(z, 64)), max(yv, broadcast(max(x, z), 64)));
        // max(broadcast(x, n), max(y, broadcast(z, n)))) -> max(y, broadcast(max(x, z), n))
        check(max(broadcast(x, 64), max(yv, broadcast(z, 64))), max(yv, broadcast(max(z, x), 64)));
        // max(broadcast(x, n), max(broadcast(y, n), z))) -> max(z, broadcast(max(x, y), n))
        check(max(broadcast(x, 64), max(broadcast(y, 64), zv)), max(zv, broadcast(max(y, x), 64)));
    }
}

void check_boolean()
{
    Expr x = Var("x"), y = Var("y"), z = Var("z"), w = Var("w");
    Expr xf = cast<float>(x);
    Expr yf = cast<float>(y);
    Expr t = const_true(), f = const_false();
    Expr b1 = Variable::make(Bool(), "b1");
    Expr b2 = Variable::make(Bool(), "b2");
    check(x == x, t);
    check(x == (x + 1), f);
    check(x - 2 == y + 3, (x - y) == 5);
    check(x + y == y + z, x == z);
    check(y + x == y + z, x == z);
    check(x + y == z + y, x == z);
    check(y + x == z + y, x == z);
    check((y + x) * 17 == (z + y) * 17, x == z);
    check(x * 0 == y * 0, t);
    check(x == x + y, y == 0);
    check(x + y == x, y == 0);
    check(100 - x == 99 - y, (y - x) == -1);

    check(x < x, f);
    check(x < (x + 1), t);
    check(x - 2 < y + 3, x < y + 5);
    check(x + y < y + z, x < z);
    check(y + x < y + z, x < z);
    check(x + y < z + y, x < z);
    check(y + x < z + y, x < z);
    check((y + x) * 17 < (z + y) * 17, x < z);
    check(x * 0 < y * 0, f);
    check(x < x + y, 0 < y);
    check(x + y < x, y < 0);

    check(select(x < 3, 2, 2), 2);
    check(select(x < (x + 1), 9, 2), 9);
    check(select(x > (x + 1), 9, 2), 2);
    // Selects of comparisons should always become selects of LT or selects of EQ
    check(select(x != 5, 2, 3), select(x == 5, 3, 2));
    check(select(x >= 5, 2, 3), select(x < 5, 3, 2));
    check(select(x <= 5, 2, 3), select(5 < x, 3, 2));
    check(select(x > 5, 2, 3), select(5 < x, 2, 3));

    check(select(x > 5, 2, 3) + select(x > 5, 6, 2), select(5 < x, 8, 5));
    check(select(x > 5, 8, 3) - select(x > 5, 6, 2), select(5 < x, 2, 1));

    check((1 - xf) * 6 < 3, 0.5f < xf);

    check(!f, t);
    check(!t, f);
    check(!(x < y), y <= x);
    check(!(x > y), x <= y);
    check(!(x >= y), x < y);
    check(!(x <= y), y < x);
    check(!(x == y), x != y);
    check(!(x != y), x == y);
    check(!(!(x == 0)), x == 0);
    check(!Expr(broadcast(x > y, 4)),
        broadcast(x <= y, 4));

    check(b1 || !b1, t);
    check(!b1 || b1, t);
    check(b1 && !b1, f);
    check(!b1 && b1, f);
    check(b1 && b1, b1);
    check(b1 || b1, b1);
    check(broadcast(b1, 4) || broadcast(!b1, 4), broadcast(t, 4));
    check(broadcast(!b1, 4) || broadcast(b1, 4), broadcast(t, 4));
    check(broadcast(b1, 4) && broadcast(!b1, 4), broadcast(f, 4));
    check(broadcast(!b1, 4) && broadcast(b1, 4), broadcast(f, 4));
    check(broadcast(b1, 4) && broadcast(b1, 4), broadcast(b1, 4));
    check(broadcast(b1, 4) || broadcast(b1, 4), broadcast(b1, 4));

    check((x == 1) && (x != 2), (x == 1));
    check((x != 1) && (x == 2), (x == 2));
    check((x == 1) && (x != 1), f);
    check((x != 1) && (x == 1), f);

    check((x == 1) || (x != 2), (x != 2));
    check((x != 1) || (x == 2), (x != 1));
    check((x == 1) || (x != 1), t);
    check((x != 1) || (x == 1), t);

    check(x < 20 || x > 19, t);
    check(x > 19 || x < 20, t);
    check(x < 20 || x > 20, x < 20 || 20 < x);
    check(x > 20 || x < 20, 20 < x || x < 20);
    check(x < 20 && x > 19, f);
    check(x > 19 && x < 20, f);
    check(x < 20 && x > 18, x < 20 && 18 < x);
    check(x > 18 && x < 20, 18 < x && x < 20);

    check(x <= 20 || x > 19, t);
    check(x > 19 || x <= 20, t);
    check(x <= 18 || x > 20, x <= 18 || 20 < x);
    check(x > 20 || x <= 18, 20 < x || x <= 18);
    check(x <= 18 && x > 19, f);
    check(x > 19 && x <= 18, f);
    check(x <= 20 && x > 19, x <= 20 && 19 < x);
    check(x > 19 && x <= 20, 19 < x && x <= 20);

    check(x < 20 || x >= 19, t);
    check(x >= 19 || x < 20, t);
    check(x < 18 || x >= 20, x < 18 || 20 <= x);
    check(x >= 20 || x < 18, 20 <= x || x < 18);
    check(x < 18 && x >= 19, f);
    check(x >= 19 && x < 18, f);
    check(x < 20 && x >= 19, x < 20 && 19 <= x);
    check(x >= 19 && x < 20, 19 <= x && x < 20);

    check(x <= 20 || x >= 21, t);
    check(x >= 21 || x <= 20, t);
    check(x <= 18 || x >= 20, x <= 18 || 20 <= x);
    check(x >= 20 || x <= 18, 20 <= x || x <= 18);
    check(x <= 18 && x >= 19, f);
    check(x >= 19 && x <= 18, f);
    check(x <= 20 && x >= 20, x <= 20 && 20 <= x);
    check(x >= 20 && x <= 20, 20 <= x && x <= 20);

    // check for substitution patterns
    check((b1 == t) && (b1 && b2), (b1 == t) && b2);
    check((b1 && b2) && (b1 == t), b2 && (b1 == t));

    {
        Expr i = Variable::make(Int(32), "i");
        check((i != 2 && (i != 4 && (i != 8 && i != 16))) || (i == 16), (i != 2 && (i != 4 && (i != 8))));
        check((i == 16) || (i != 2 && (i != 4 && (i != 8 && i != 16))), (i != 2 && (i != 4 && (i != 8))));
    }

    check(t && (x < 0), x < 0);
    check(f && (x < 0), f);
    check(t || (x < 0), t);
    check(f || (x < 0), x < 0);

    check(x == y || y != x, t);
    check(x == y || x != y, t);
    check(x == y && x != y, f);
    check(x == y && y != x, f);
    check(x < y || x >= y, t);
    check(x <= y || x > y, t);
    check(x < y && x >= y, f);
    check(x <= y && x > y, f);

    check(x <= max(x, y), t);
    check(x < min(x, y), f);
    check(min(x, y) <= x, t);
    check(max(x, y) < x, f);
    check(max(x, y) <= y, x <= y);
    check(min(x, y) >= y, y <= x);

    check((1 < y) && (2 < y), 2 < y);

    check(x * 5 < 4, x < 1);
    check(x * 5 < 5, x < 1);
    check(x * 5 < 6, x < 2);
    check(x * 5 <= 4, x <= 0);
    check(x * 5 <= 5, x <= 1);
    check(x * 5 <= 6, x <= 1);
    check(x * 5 > 4, 0 < x);
    check(x * 5 > 5, 1 < x);
    check(x * 5 > 6, 1 < x);
    check(x * 5 >= 4, 1 <= x);
    check(x * 5 >= 5, 1 <= x);
    check(x * 5 >= 6, 2 <= x);

    check(x / 4 < 3, x < 12);
    check(3 < x / 4, 15 < x);

    check(4 - x <= 0, 4 <= x);

    check((x / 8) * 8 < x - 8, f);
    check((x / 8) * 8 < x - 9, f);
    check((x / 8) * 8 < x - 7, f);
    check((x / 8) * 8 < x - 6, 6 < x % 8);
    check(ramp(x * 4, 1, 4) < broadcast(y * 4, 4), broadcast(x < y, 4));
    check(ramp(x * 8, 1, 4) < broadcast(y * 8, 4), broadcast(x < y, 4));
    check(ramp(x * 8 + 1, 1, 4) < broadcast(y * 8, 4), broadcast(x < y, 4));
    check(ramp(x * 8 + 4, 1, 4) < broadcast(y * 8, 4), broadcast(x < y, 4));
    check(ramp(x * 8 + 8, 1, 4) < broadcast(y * 8, 4), broadcast(x < y + (-1), 4));
    check(ramp(x * 8 + 5, 1, 4) < broadcast(y * 8, 4), ramp(x * 8 + 5, 1, 4) < broadcast(y * 8, 4));
    check(ramp(x * 8 - 1, 1, 4) < broadcast(y * 8, 4), ramp(x * 8 + (-1), 1, 4) < broadcast(y * 8, 4));
    check(ramp(x * 8, 1, 4) < broadcast(y * 4, 4), broadcast(x * 2 < y, 4));
    check(ramp(x * 8, 2, 4) < broadcast(y * 8, 4), broadcast(x < y, 4));
    check(ramp(x * 8 + 1, 2, 4) < broadcast(y * 8, 4), broadcast(x < y, 4));
    check(ramp(x * 8 + 2, 2, 4) < broadcast(y * 8, 4), ramp(x * 8 + 2, 2, 4) < broadcast(y * 8, 4));
    check(ramp(x * 8, 3, 4) < broadcast(y * 8, 4), ramp(x * 8, 3, 4) < broadcast(y * 8, 4));
    check(select(ramp((x / 16) * 16, 1, 8) < broadcast((y / 8) * 8, 8), broadcast(1, 8), broadcast(3, 8)),
        select((x / 16) * 2 < y / 8, broadcast(1, 8), broadcast(3, 8)));

    check(ramp(x * 8, -1, 4) < broadcast(y * 8, 4), ramp(x * 8, -1, 4) < broadcast(y * 8, 4));
    check(ramp(x * 8 + 1, -1, 4) < broadcast(y * 8, 4), ramp(x * 8 + 1, -1, 4) < broadcast(y * 8, 4));
    check(ramp(x * 8 + 4, -1, 4) < broadcast(y * 8, 4), broadcast(x < y, 4));
    check(ramp(x * 8 + 8, -1, 4) < broadcast(y * 8, 4), ramp(x * 8 + 8, -1, 4) < broadcast(y * 8, 4));
    check(ramp(x * 8 + 5, -1, 4) < broadcast(y * 8, 4), broadcast(x < y, 4));
    check(ramp(x * 8 - 1, -1, 4) < broadcast(y * 8, 4), broadcast(x < y + 1, 4));

    // Check anded conditions apply to the then case only
    check(IfThenElse::make(x == 4 && y == 5,
              Evaluate::make(z + x + y),
              Evaluate::make(z + x - y)),
        IfThenElse::make(x == 4 && y == 5,
            Evaluate::make(z + 9),
            Evaluate::make(z + x - y)));

    // Check ored conditions apply to the else case only
    check(IfThenElse::make(b1 || b2,
              Evaluate::make(select(b1, x + 3, y + 4) + select(b2, x + 5, y + 7)),
              Evaluate::make(select(b1, x + 3, y + 8) - select(b2, x + 5, y + 7))),
        IfThenElse::make(b1 || b2,
            Evaluate::make(select(b1, x + 3, y + 4) + select(b2, x + 5, y + 7)),
            Evaluate::make(1)));

    // Check single conditions apply to both cases of an ifthenelse
    check(IfThenElse::make(b1,
              Evaluate::make(select(b1, x, y)),
              Evaluate::make(select(b1, z, w))),
        IfThenElse::make(b1,
            Evaluate::make(x),
            Evaluate::make(w)));

    check(IfThenElse::make(x < y,
              IfThenElse::make(x < y, Evaluate::make(y), Evaluate::make(x)),
              Evaluate::make(x)),
        IfThenElse::make(x < y,
            Evaluate::make(y),
            Evaluate::make(x)));

    check(Block::make(IfThenElse::make(x < y, Evaluate::make(x + 1), Evaluate::make(x + 2)),
              IfThenElse::make(x < y, Evaluate::make(x + 3), Evaluate::make(x + 4))),
        IfThenElse::make(x < y,
            Block::make(Evaluate::make(x + 1), Evaluate::make(x + 3)),
            Block::make(Evaluate::make(x + 2), Evaluate::make(x + 4))));

    check(Block::make(IfThenElse::make(x < y, Evaluate::make(x + 1)),
              IfThenElse::make(x < y, Evaluate::make(x + 2))),
        IfThenElse::make(x < y, Block::make(Evaluate::make(x + 1), Evaluate::make(x + 2))));

    check(Block::make(IfThenElse::make(x < y, Evaluate::make(x + 1), Evaluate::make(x + 2)),
              IfThenElse::make(x < y, Evaluate::make(x + 3))),
        IfThenElse::make(x < y,
            Block::make(Evaluate::make(x + 1), Evaluate::make(x + 3)),
            Evaluate::make(x + 2)));

    check(Block::make(IfThenElse::make(x < y, Evaluate::make(x + 1)),
              IfThenElse::make(x < y, Evaluate::make(x + 2), Evaluate::make(x + 3))),
        IfThenElse::make(x < y,
            Block::make(Evaluate::make(x + 1), Evaluate::make(x + 2)),
            Evaluate::make(x + 3)));

    // Check conditions involving entire exprs
    Expr foo = x + 3 * y;
    Expr foo_simple = x + y * 3;
    check(IfThenElse::make(foo == 17,
              Evaluate::make(x + foo + 1),
              Evaluate::make(x + foo + 2)),
        IfThenElse::make(foo_simple == 17,
            Evaluate::make(x + 18),
            Evaluate::make(x + foo_simple + 2)));

    check(IfThenElse::make(foo != 17,
              Evaluate::make(x + foo + 1),
              Evaluate::make(x + foo + 2)),
        IfThenElse::make(foo_simple != 17,
            Evaluate::make(x + foo_simple + 1),
            Evaluate::make(x + 19)));

    // The construct
    //     if (var == expr) then a else b;
    // was being simplified incorrectly, but *only* if var was of type Bool.
    Stmt then_clause = AssertStmt::make(b2, Expr(22), Evaluate::make(0));
    Stmt else_clause = AssertStmt::make(b2, Expr(33), Evaluate::make(0));
    check(IfThenElse::make(b1 == b2, then_clause, else_clause),
        IfThenElse::make(b1 == b2, then_clause, else_clause));

    // Simplifications of selects
    check(select(x == 3, 5, 7) + 7, select(x == 3, 12, 14));
    check(select(x == 3, 5, 7) - 7, select(x == 3, -2, 0));
    check(select(x == 3, 5, y) - y, select(x == 3, 5 - y, 0));
    check(select(x == 3, y, 5) - y, select(x == 3, 0, 5 - y));
    check(y - select(x == 3, 5, y), select(x == 3, y + (-5), 0));
    check(y - select(x == 3, y, 5), select(x == 3, 0, y + (-5)));

    check(select(x == 3, 5, 7) == 7, x != 3);
    check(select(x == 3, z, y) == z, (x == 3) || (y == z));

    check(select(x == 3, 4, 2) == 0, const_false());
    check(select(x == 3, y, 2) == 4, (x == 3) && (y == 4));
    check(select(x == 3, 2, y) == 4, (x != 3) && (y == 4));

    check(min(select(x == 2, y * 3, 8), select(x == 2, y + 8, y * 7)),
        select(x == 2, min(y * 3, y + 8), min(y * 7, 8)));

    check(max(select(x == 2, y * 3, 8), select(x == 2, y + 8, y * 7)),
        select(x == 2, max(y * 3, y + 8), max(y * 7, 8)));

    check(select(x == 2, x + 1, x + 5), x + select(x == 2, 1, 5));
    check(select(x == 2, x + y, x + z), x + select(x == 2, y, z));
    check(select(x == 2, y + x, x + z), x + select(x == 2, y, z));
    check(select(x == 2, y + x, z + x), select(x == 2, y, z) + x);
    check(select(x == 2, x + y, z + x), x + select(x == 2, y, z));
    check(select(x == 2, x * 2, x * 5), x * select(x == 2, 2, 5));
    check(select(x == 2, x * y, x * z), x * select(x == 2, y, z));
    check(select(x == 2, y * x, x * z), x * select(x == 2, y, z));
    check(select(x == 2, y * x, z * x), select(x == 2, y, z) * x);
    check(select(x == 2, x * y, z * x), x * select(x == 2, y, z));
    check(select(x == 2, x - y, x - z), x - select(x == 2, y, z));
    check(select(x == 2, y - x, z - x), select(x == 2, y, z) - x);
    check(select(x == 2, x + y, x - z), x + select(x == 2, y, 0 - z));
    check(select(x == 2, y + x, x - z), x + select(x == 2, y, 0 - z));
    check(select(x == 2, x - z, x + y), x + select(x == 2, 0 - z, y));
    check(select(x == 2, x - z, y + x), x + select(x == 2, 0 - z, y));

    {

        Expr b[12];
        for (int i = 0; i < 12; i++) {
            b[i] = Variable::make(Bool(), "b");
        }

        // Some rules that collapse selects
        check(select(b[0], x, select(b[1], x, y)),
            select(b[0] || b[1], x, y));
        check(select(b[0], x, select(b[1], y, x)),
            select(b[0] || !b[1], x, y));
        check(select(b[0], select(b[1], x, y), x),
            select(b[0] && !b[1], y, x));
        check(select(b[0], select(b[1], y, x), x),
            select(b[0] && b[1], y, x));

        // Ternary boolean expressions in two variables
        check(b[0] || (b[0] && b[1]), b[0]);
        check((b[0] && b[1]) || b[0], b[0]);
        check(b[0] && (b[0] || b[1]), b[0]);
        check((b[0] || b[1]) && b[0], b[0]);
        check(b[0] && (b[0] && b[1]), b[0] && b[1]);
        check((b[0] && b[1]) && b[0], b[1] && b[0]);
        check(b[0] || (b[0] || b[1]), b[0] || b[1]);
        check((b[0] || b[1]) || b[0], b[1] || b[0]);

        // A nasty unsimplified boolean Expr seen in the wild
        Expr nasty = ((((((((((((((((((((((((((((((((((((((((((((b[0] && b[1]) || (b[2] && b[1])) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[6]) || (b[2] && b[6]))) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[3]) || (b[2] && b[3]))) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[7]) || (b[2] && b[7]))) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[4]) || (b[2] && b[4]))) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[8]) || (b[2] && b[8]))) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[5]) || (b[2] && b[5]))) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[10]) || (b[2] && b[10]))) || b[0]) || b[2]) || b[0]) || b[2]) && ((b[0] && b[9]) || (b[2] && b[9]))) || b[0]) || b[2]);
        check(nasty, b[0] || b[2]);
    }
}

void check_math()
{
    Expr x = Var("x");

    check(sqrt(4.0f), 2.0f);
    check(log(0.5f + 0.5f), 0.0f);
    check(exp(log(2.0f)), 2.0f);
    check(pow(4.0f, 0.5f), 2.0f);
    check(round(1000.0f * pow(exp(1.0f), log(10.0f))), 10000.0f);

    check(floor(0.98f), 0.0f);
    check(ceil(0.98f), 1.0f);
    check(round(0.6f), 1.0f);
    check(round(-0.5f), 0.0f);
    check(trunc(-1.6f), -1.0f);
    check(floor(round(x)), round(x));
    check(ceil(ceil(x)), ceil(x));
}

void check_overflow()
{
    Expr overflowing[] = {
        make_const(Int(32), 0x7fffffff) + 1,
        make_const(Int(32), 0x7ffffff0) + 16,
        (make_const(Int(32), 0x7fffffff) + make_const(Int(32), 0x7fffffff)),
        make_const(Int(32), 0x08000000) * 16,
        (make_const(Int(32), 0x00ffffff) * make_const(Int(32), 0x00ffffff)),
        make_const(Int(32), 0x80000000) - 1,
        0 - make_const(Int(32), 0x80000000),
        make_const(Int(64), (int64_t)0x7fffffffffffffffLL) + 1,
        make_const(Int(64), (int64_t)0x7ffffffffffffff0LL) + 16,
        (make_const(Int(64), (int64_t)0x7fffffffffffffffLL) + make_const(Int(64), (int64_t)0x7fffffffffffffffLL)),
        make_const(Int(64), (int64_t)0x0800000000000000LL) * 16,
        (make_const(Int(64), (int64_t)0x00ffffffffffffffLL) * make_const(Int(64), (int64_t)0x00ffffffffffffffLL)),
        make_const(Int(64), (int64_t)0x8000000000000000LL) - 1,
        0 - make_const(Int(64), (int64_t)0x8000000000000000LL),
    };
    Expr not_overflowing[] = {
        make_const(Int(32), 0x7ffffffe) + 1,
        make_const(Int(32), 0x7fffffef) + 16,
        make_const(Int(32), 0x07ffffff) * 2,
        (make_const(Int(32), 0x0000ffff) * make_const(Int(32), 0x00008000)),
        make_const(Int(32), 0x80000001) - 1,
        0 - make_const(Int(32), 0x7fffffff),
        make_const(Int(64), (int64_t)0x7ffffffffffffffeLL) + 1,
        make_const(Int(64), (int64_t)0x7fffffffffffffefLL) + 16,
        make_const(Int(64), (int64_t)0x07ffffffffffffffLL) * 16,
        (make_const(Int(64), (int64_t)0x00000000ffffffffLL) * make_const(Int(64), (int64_t)0x0000000080000000LL)),
        make_const(Int(64), (int64_t)0x8000000000000001LL) - 1,
        0 - make_const(Int(64), (int64_t)0x7fffffffffffffffLL),
    };

    for (Expr e : overflowing) {
        internal_assert(!is_const(simplify(e)))
            << "Overflowing expression should not have simplified: " << e << "\n";
    }
    for (Expr e : not_overflowing) {
        internal_assert(is_const(simplify(e)))
            << "Non-everflowing expression should have simplified: " << e << "\n";
    }
}

void check_ind_expr(Expr e, bool expect_error)
{
    Expr e2 = simplify(e);
    const Call* call = e2.as<Call>();
    bool is_error = call && call->is_intrinsic(Call::indeterminate_expression);
    if (expect_error && !is_error)
        internal_error << "Expression should be indeterminate: " << e << " but saw: " << e2 << "\n";
    else if (!expect_error && is_error)
        internal_error << "Expression should not be indeterminate: " << e << " but saw: " << e2 << "\n";
}

void check_indeterminate_ops(Expr e, bool e_is_zero, bool e_is_indeterminate)
{
    Expr b = cast<bool>(e);
    Expr t = const_true(), f = const_false();
    Expr one = cast(e.type(), 1);
    Expr zero = cast(e.type(), 0);

    check_ind_expr(e, e_is_indeterminate);
    check_ind_expr(e + e, e_is_indeterminate);
    check_ind_expr(e - e, e_is_indeterminate);
    check_ind_expr(e * e, e_is_indeterminate);
    check_ind_expr(e / e, e_is_zero || e_is_indeterminate);
    check_ind_expr((1 / e) / e, e_is_zero || e_is_indeterminate);
    // Expr::operator% asserts if denom is constant zero.
    if (!is_zero(e)) {
        check_ind_expr(e % e, e_is_zero || e_is_indeterminate);
        check_ind_expr((1 / e) % e, e_is_zero || e_is_indeterminate);
    }
    check_ind_expr(min(e, one), e_is_indeterminate);
    check_ind_expr(max(e, one), e_is_indeterminate);
    check_ind_expr(e == one, e_is_indeterminate);
    check_ind_expr(one == e, e_is_indeterminate);
    check_ind_expr(e < one, e_is_indeterminate);
    check_ind_expr(one < e, e_is_indeterminate);
    check_ind_expr(!(e == one), e_is_indeterminate);
    check_ind_expr(!(one == e), e_is_indeterminate);
    check_ind_expr(!(e < one), e_is_indeterminate);
    check_ind_expr(!(one < e), e_is_indeterminate);
    check_ind_expr(b && t, e_is_indeterminate);
    check_ind_expr(t && b, e_is_indeterminate);
    check_ind_expr(b || t, e_is_indeterminate);
    check_ind_expr(t || b, e_is_indeterminate);
    check_ind_expr(!b, e_is_indeterminate);
    check_ind_expr(select(b, one, zero), e_is_indeterminate);
    check_ind_expr(select(t, e, zero), e_is_indeterminate);
    check_ind_expr(select(f, zero, e), e_is_indeterminate);
    check_ind_expr(e << one, e_is_indeterminate);
    check_ind_expr(e >> one, e_is_indeterminate);
    // Avoid warnings for things like (1 << 2147483647)
    if (e_is_indeterminate) {
        check_ind_expr(one << e, e_is_indeterminate);
        check_ind_expr(one >> e, e_is_indeterminate);
    }
    check_ind_expr(one & e, e_is_indeterminate);
    check_ind_expr(e & one, e_is_indeterminate);
    check_ind_expr(one | e, e_is_indeterminate);
    check_ind_expr(e | one, e_is_indeterminate);
    if (!e.type().is_uint()) {
        // Avoid warnings
        check_ind_expr(abs(e), e_is_indeterminate);
    }
    check_ind_expr(log(e), e_is_indeterminate);
    check_ind_expr(sqrt(e), e_is_indeterminate);
    check_ind_expr(exp(e), e_is_indeterminate);
    check_ind_expr(pow(e, one), e_is_indeterminate);
    // pow(x, y) explodes for huge integer y (Issue #1441)
    if (e_is_indeterminate) {
        check_ind_expr(pow(one, e), e_is_indeterminate);
    }
    check_ind_expr(floor(e), e_is_indeterminate);
    check_ind_expr(ceil(e), e_is_indeterminate);
    check_ind_expr(round(e), e_is_indeterminate);
    check_ind_expr(trunc(e), e_is_indeterminate);
}

void check_indeterminate()
{
    const int32_t values[] = {
        int32_t(0x80000000),
        -2147483647,
        -2,
        -1,
        0,
        1,
        2,
        2147483647,
    };

    for (int32_t i1 : values) {
        // reality-check for never-indeterminate values.
        check_indeterminate_ops(Expr(i1), !i1, false);
        for (int32_t i2 : values) {
            {
                Expr e1(i1), e2(i2);
                Expr r = (e1 / e2);
                bool r_is_zero = !i1 || (i2 != 0 && !div_imp((int64_t)i1, (int64_t)i2)); // avoid trap for -2147483648/-1
                bool r_is_ind = !i2;
                check_indeterminate_ops(r, r_is_zero, r_is_ind);

                // Expr::operator% asserts if denom is constant zero.
                if (!is_zero(e2)) {
                    Expr m = (e1 % e2);
                    bool m_is_zero = !i1 || (i2 != 0 && !mod_imp((int64_t)i1, (int64_t)i2)); // avoid trap for -2147483648/-1
                    bool m_is_ind = !i2;
                    check_indeterminate_ops(m, m_is_zero, m_is_ind);
                }
            }
            {
                uint32_t u1 = (uint32_t)i1;
                uint32_t u2 = (uint32_t)i2;
                Expr e1(u1), e2(u2);
                Expr r = (e1 / e2);
                bool r_is_zero = !u1 || (u2 != 0 && !div_imp(u1, u2));
                bool r_is_ind = !u2;
                check_indeterminate_ops(r, r_is_zero, r_is_ind);

                // Expr::operator% asserts if denom is constant zero.
                if (!is_zero(e2)) {
                    Expr m = (e1 % e2);
                    bool m_is_zero = !u1 || (u2 != 0 && !mod_imp(u1, u2));
                    bool m_is_ind = !u2;
                    check_indeterminate_ops(m, m_is_zero, m_is_ind);
                }
            }
        }
    }
}

// void simplify_test() {
TEST(IRSIMPLIFY, Basic)
{
    VarExpr x = Var("x"), y = Var("y"), z = Var("z"), w = Var("w"), v = Var("v");
    Expr xf = cast<float>(x);
    Expr yf = cast<float>(y);
    Expr t = const_true(), f = const_false();
    check_indeterminate();
    check_casts();
    check_algebra();
    check_vectors();
    check_bounds();
    check_math();
    check_boolean();
    check_overflow();

    // Check bitshift operations
    //check(cast(Int(16), x) << 10, cast(Int(16), x) * 1024);
    //check(cast(Int(16), x) >> 10, cast(Int(16), x) / 1024);
    //check(cast(Int(16), x) << -10, cast(Int(16), x) / 1024);
    // Correctly triggers a warning:
    //check(cast(Int(16), x) << 20, cast(Int(16), x) << 20);

    // Check bitwise_and. (Added as result of a bug.)
    // TODO: more coverage of bitwise_and and bitwise_or.
    check(cast(UInt(32), x) & Expr((uint32_t)0xaaaaaaaa),
        cast(UInt(32), x) & Expr((uint32_t)0xaaaaaaaa));

    // Check that chains of widening casts don't lose the distinction
    // between zero-extending and sign-extending.
    check(cast(UInt(64), cast(UInt(32), cast(Int(8), -1))),
        UIntImm::make(UInt(64), 0xffffffffULL));

    v = Variable::make(Int(32, 4), "v");
    // Check constants get pushed inwards
    check(Let::make(x, 3, x + 4), 7);

    // Check ramps in lets get pushed inwards
    check(Let::make(v, ramp(x * 2 + 7, 3, 4), v + Expr(broadcast(2, 4))),
        ramp(x * 2 + 9, 3, 4));

    // Check broadcasts in lets get pushed inwards
    check(Let::make(v, broadcast(x, 4), v + Expr(broadcast(2, 4))),
        broadcast(x + 2, 4));

    // Check that dead lets get stripped
    check(Let::make(x, 3 * y * y * y, 4), 4);
    check(Let::make(x, 0, 0), 0);

    // Check that lets inside an evaluate node get lifted
    check(Evaluate::make(Let::make(x, Call::make(Int(32), "dummy", { 3, x, 4 }, Call::Extern), Let::make(y, 10, x + y + 2))),
        LetStmt::make(x, Call::make(Int(32), "dummy", { 3, x, 4 }, Call::Extern), Evaluate::make(x + 12)));

    // Test case with most negative 32-bit number, as constant to check that it is not negated.
    check(((x * (int32_t)0x80000000) + (y + z * (int32_t)0x80000000)),
        ((x * (int32_t)0x80000000) + (y + z * (int32_t)0x80000000)));

    // Check that constant args to a stringify get combined
    check(Call::make(type_of<const char*>(), Call::stringify, { 3, string(" "), 4 }, Call::Intrinsic),
        string("3 4"));

    check(Call::make(type_of<const char*>(), Call::stringify, { 3, x, 4, string(", "), 3.4f }, Call::Intrinsic),
        Call::make(type_of<const char*>(), Call::stringify, { string("3"), x, string("4, 3.400000") }, Call::Intrinsic));

    // Check min(x, y)*max(x, y) gets simplified into x*y
    check(min(x, y) * max(x, y), x * y);
    check(min(x, y) * max(y, x), x * y);
    check(max(x, y) * min(x, y), x * y);
    check(max(y, x) * min(x, y), x * y);

    // Check min(x, y) + max(x, y) gets simplified into x + y
    check(min(x, y) + max(x, y), x + y);
    check(min(x, y) + max(y, x), x + y);
    check(max(x, y) + min(x, y), x + y);
    check(max(y, x) + min(x, y), x + y);

    // Check max(min(x, y), max(x, y)) gets simplified into max(x, y)
    check(max(min(x, y), max(x, y)), max(x, y));
    check(max(min(x, y), max(y, x)), max(x, y));
    check(max(max(x, y), min(x, y)), max(x, y));
    check(max(max(y, x), min(x, y)), max(x, y));

    // Check min(max(x, y), min(x, y)) gets simplified into min(x, y)
    check(min(max(x, y), min(x, y)), min(x, y));
    check(min(max(x, y), min(y, x)), min(x, y));
    check(min(min(x, y), max(x, y)), min(x, y));
    check(min(min(y, x), max(x, y)), min(x, y));

    // Check if we can simplify away comparison on vector types considering bounds.
    Scope<Interval> bounds_info;
    bounds_info.push(x.get(), Interval(0, 4));
    check_in_bounds(ramp(x, 1, 4) < broadcast(0, 4), const_false(4), bounds_info);
    check_in_bounds(ramp(x, 1, 4) < broadcast(8, 4), const_true(4), bounds_info);
    check_in_bounds(ramp(x, -1, 4) < broadcast(-4, 4), const_false(4), bounds_info);
    check_in_bounds(ramp(x, -1, 4) < broadcast(5, 4), const_true(4), bounds_info);
    check_in_bounds(min(ramp(x, 1, 4), broadcast(0, 4)), broadcast(0, 4), bounds_info);
    check_in_bounds(min(ramp(x, 1, 4), broadcast(8, 4)), ramp(x, 1, 4), bounds_info);
    check_in_bounds(min(ramp(x, -1, 4), broadcast(-4, 4)), broadcast(-4, 4), bounds_info);
    check_in_bounds(min(ramp(x, -1, 4), broadcast(5, 4)), ramp(x, -1, 4), bounds_info);
    check_in_bounds(max(ramp(x, 1, 4), broadcast(0, 4)), ramp(x, 1, 4), bounds_info);
    check_in_bounds(max(ramp(x, 1, 4), broadcast(8, 4)), broadcast(8, 4), bounds_info);
    check_in_bounds(max(ramp(x, -1, 4), broadcast(-4, 4)), ramp(x, -1, 4), bounds_info);
    check_in_bounds(max(ramp(x, -1, 4), broadcast(5, 4)), broadcast(5, 4), bounds_info);

    // Collapse some vector interleaves
    check(interleave_vectors({ ramp(x, 2, 4), ramp(x + 1, 2, 4) }), ramp(x, 1, 8));
    check(interleave_vectors({ ramp(x, 4, 4), ramp(x + 2, 4, 4) }), ramp(x, 2, 8));
    check(interleave_vectors({ ramp(x - y, 2 * y, 4), ramp(x, 2 * y, 4) }), ramp(x - y, y, 8));
    check(interleave_vectors({ ramp(x, 3, 4), ramp(x + 1, 3, 4), ramp(x + 2, 3, 4) }), ramp(x, 1, 12));
    {
        Expr vec = ramp(x, 1, 16);
        check(interleave_vectors({ slice(vec, 0, 2, 8), slice(vec, 1, 2, 8) }), vec);
        check(interleave_vectors({ slice(vec, 0, 4, 4), slice(vec, 1, 4, 4), slice(vec, 2, 4, 4), slice(vec, 3, 4, 4) }), vec);
    }

    // Collapse some vector concats
    check(concat_vectors({ ramp(x, 2, 4), ramp(x + 8, 2, 4) }), ramp(x, 2, 8));
    check(concat_vectors({ ramp(x, 3, 2), ramp(x + 6, 3, 2), ramp(x + 12, 3, 2) }), ramp(x, 3, 6));

    // Now some ones that can't work
    {
        Expr e = interleave_vectors({ ramp(x, 2, 4), ramp(x, 2, 4) });
        check(e, e);
        e = interleave_vectors({ ramp(x, 2, 4), ramp(x + 2, 2, 4) });
        check(e, e);
        e = interleave_vectors({ ramp(x, 3, 4), ramp(x + 1, 3, 4) });
        check(e, e);
        e = interleave_vectors({ ramp(x, 2, 4), ramp(y + 1, 2, 4) });
        check(e, e);
        e = interleave_vectors({ ramp(x, 2, 4), ramp(x + 1, 3, 4) });
        check(e, e);

        e = concat_vectors({ ramp(x, 1, 4), ramp(x + 4, 2, 4) });
        check(e, e);
        e = concat_vectors({ ramp(x, 1, 4), ramp(x + 8, 1, 4) });
        check(e, e);
        e = concat_vectors({ ramp(x, 1, 4), ramp(y + 4, 1, 4) });
        check(e, e);
    }

    // Now check that an interleave of some collapsible loads collapses into a single dense load
    {
        VarExpr buf = Var("buf"), buf2 = Var("buf2");
        Expr load1 = Load::make(Float(32, 4), buf, ramp(x, 2, 4), const_true(4));
        Expr load2 = Load::make(Float(32, 4), buf, ramp(x + 1, 2, 4), const_true(4));
        Expr load12 = Load::make(Float(32, 8), buf, ramp(x, 1, 8), const_true(8));
        check(interleave_vectors({ load1, load2 }), load12);

        // They don't collapse in the other order
        Expr e = interleave_vectors({ load2, load1 });
        check(e, e);

        // Or if the buffers are different
        Expr load3 = Load::make(Float(32, 4), buf2, ramp(x + 1, 2, 4), const_true(4));
        e = interleave_vectors({ load1, load3 });
        check(e, e);
    }

    // Check that concatenated loads of adjacent scalars collapse into a vector load.
    {
        VarExpr buf = Var("buf");
        int lanes = 4;
        std::vector<Expr> loads;
        for (int i = 0; i < lanes; i++) {
            loads.push_back(Load::make(Float(32), buf, x + i, const_true()));
        }

        check(concat_vectors(loads), Load::make(Float(32, lanes), buf, ramp(x, 1, lanes), const_true(lanes)));
    }

    // This expression doesn't simplify, but it did cause exponential
    // slowdown at one stage.
    {
        Expr e = x;
        for (int i = 0; i < 100; i++) {
            e = max(e, 1) / 2;
        }
        check(e, e);
    }

    // This expression is used to cause infinite recursion.
    {
        Expr e = Broadcast::make(-16, 2) < (ramp(Cast::make(UInt(16), 7), Cast::make(UInt(16), 11), 2) - Broadcast::make(1, 2));
        Expr expected = Broadcast::make(-16, 2) < (ramp(make_const(UInt(16), 7), make_const(UInt(16), 11), 2) - Broadcast::make(1, 2));
        check(e, expected);
    }

    {

        VarExpr f = Var("f");
        Expr pred = ramp(x * y + x * z, 2, 8) > 2;
        Expr index = ramp(x + y, 1, 8);
        Expr value = Load::make(index.type(), f, index, const_true(index.type().lanes()));
        Stmt stmt = Store::make(f, value, index, pred);
        check(stmt, Evaluate::make(0));
    }

    {
        // Verify that integer types passed to min() and max() are coerced to match
        // Exprs, rather than being promoted to int first. (TODO: This doesn't really
        // belong in the test for Simplify, but IROperator has no test unit of its own.)
        Expr one = cast<uint16_t>(1);
        const int two = 2; // note that type is int, not uint16_t
        Expr r1, r2, r3;

        r1 = min(one, two);
        internal_assert(r1.type() == BaseTypeCast<uint16_t>());

        // TD<decltype(HalideIR::type_of<uint16_t>())> oneType;

        r2 = min(one, two, one);
        internal_assert(r2.type() == BaseTypeCast<uint16_t>());
        // Explicitly passing 'two' as an Expr, rather than an int, will defeat this logic.
        r3 = min(one, Expr(two), one);
        internal_assert(r3.type() == BaseTypeCast<int>());

        r1 = max(one, two);
        internal_assert(r1.type() == BaseTypeCast<uint16_t>());
        r2 = max(one, two, one);
        internal_assert(r2.type() == BaseTypeCast<uint16_t>());
        // Explicitly passing 'two' as an Expr, rather than an int, will defeat this logic.
        r3 = max(one, Expr(two), one);
        internal_assert(r3.type() == BaseTypeCast<int>());
    }

    {
        Expr x = Variable::make(UInt(32), "x");
        Expr y = Variable::make(UInt(32), "y");
        // This is used to get simplified into broadcast(x - y, 2) which is
        // incorrect when there is overflow.
        Expr e = simplify(max(ramp(x, y, 2), broadcast(x, 2)) - max(broadcast(y, 2), ramp(y, y, 2)));
        Expr expected = max(ramp(x, y, 2), broadcast(x, 2)) - max(ramp(y, y, 2), broadcast(y, 2));
        check(e, expected);
    }

    std::cout << "Simplify test passed" << std::endl;
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();
}
