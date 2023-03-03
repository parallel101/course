#pragma once

// Based on https://www.codenong.com/6707148/
// With some tiny usage tweaks
/*
 * The PP_NARG macro evaluates to the number of arguments that have been
 * passed to it.
 *
 * Laurent Deniau,"__VA_NARG__," 17 January 2006, <comp.std.c> (29 November 2007).
 */
#define PP_NARG(...)    PP_NARG_(__VA_ARGS__,PP_RSEQ_N())
#define PP_NARG_(...)   PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N( \
        _1, _2, _3, _4, _5, _6, _7, _8, _9,_10,  \
        _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
        _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
        _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
        _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
        _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
        _61,_62,_63,N,...) N
#define PP_RSEQ_N() \
        63,62,61,60,                   \
        59,58,57,56,55,54,53,52,51,50, \
        49,48,47,46,45,44,43,42,41,40, \
        39,38,37,36,35,34,33,32,31,30, \
        29,28,27,26,25,24,23,22,21,20, \
        19,18,17,16,15,14,13,12,11,10, \
        9,8,7,6,5,4,3,2,1,0
/* need extra level to force extra eval */
#define PP_PASTEX(a,b) a ## b
#define PP_XPASTE(a,b) PP_PASTEX(a,b)
/* PP_APPLYXn variadic X-Macro by M Joshua Ryan   */
/* Free for all uses. Don't be a jerk.            */
/* I got bored after typing 15 of these.          */
/* You could keep going upto 64 (PPNARG's limit). */
#define PP_APPLYX1(X,_,a)           X(a)
#define PP_APPLYX2(X,_,a,b)         X(a)_ X(b)
#define PP_APPLYX3(X,_,a,b,c)       X(a)_ X(b)_ X(c)
#define PP_APPLYX4(X,_,a,b,c,d)     X(a)_ X(b)_ X(c)_ X(d)
#define PP_APPLYX5(X,_,a,b,c,d,e)   X(a)_ X(b)_ X(c)_ X(d)_ X(e)
#define PP_APPLYX6(X,_,a,b,c,d,e,f)_ X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)
#define PP_APPLYX7(X,_,a,b,c,d,e,f,g) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)
#define PP_APPLYX8(X,_,a,b,c,d,e,f,g,h) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)
#define PP_APPLYX9(X,_,a,b,c,d,e,f,g,h,i) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)_ X(i)
#define PP_APPLYX10(X,_,a,b,c,d,e,f,g,h,i,j) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)_ X(i)_ X(j)
#define PP_APPLYX11(X,_,a,b,c,d,e,f,g,h,i,j,k) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)_ X(i)_ X(j)_ X(k)
#define PP_APPLYX12(X,_,a,b,c,d,e,f,g,h,i,j,k,l) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)_ X(i)_ X(j)_ X(k)_ X(l)
#define PP_APPLYX13(X,_,a,b,c,d,e,f,g,h,i,j,k,l,m) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)_ X(i)_ X(j)_ X(k)_ X(l)_ X(m)
#define PP_APPLYX14(X,_,a,b,c,d,e,f,g,h,i,j,k,l,m,n) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)_ X(i)_ X(j)_ X(k)_ X(l)_ X(m)_ X(n)
#define PP_APPLYX15(X,_,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) \
    X(a)_ X(b)_ X(c)_ X(d)_ X(e)_ X(f)_ X(g)_ X(h)_ X(i)_ X(j)_ X(k)_ X(l)_ X(m)_ X(n)_ X(o)
#define PP_APPLYX_(X,_,M, ...) M(X,_,__VA_ARGS__)
#define PP_APPLYXn(X,_,...) PP_APPLYX_(X,_,PP_XPASTE(PP_APPLYX, PP_NARG(__VA_ARGS__)), __VA_ARGS__)
#define PP_FOREACH(func, join, ...) PP_APPLYXn(func, join, __VA_ARGS__)
