package cetus.hir;

import java.util.*;

/**
* Repository for C99 standard library functions. This class provides a basic
* information about the standard library calls. Knowing that if a function call
* may or must not have side effects can greatly improve the precision of a
* program analysis in general. Standard library calls may have side effects on
* the parameters, on the automatic storage locations, on the file system, and
* on the execution environment.
*/
public class StandardLibrary {

    /** Only a single object is constructed. */
    private static final StandardLibrary std = new StandardLibrary();

    /** Predefined properties for each library functions */
    private Map<String, Set<Property>> catalog;

    private static Map<String, int[]> seIndices = new HashMap<String, int[]>();

    /** Predefined set of properties */
    private enum Property {
        SIDE_EFFECT_GLOBAL,     // contains side effects on global variables.
        SIDE_EFFECT_PARAMETER,  // contains side effects through parameters.
        SIDE_EFFECT_STDIO,      // the only side effect is on standard io.
        SIDE_EFFECT_FILEIO,     // the only side effect is on file io.
        MAY_NOT_RETURN          // it is possible for the call not to return.
    }

    /**
    * Checks if the given function call is a standard library call.
    * @param fcall the function call to be examined.
    * @return true if the function call exists in the entries.
    */
    public static boolean contains(FunctionCall fcall) {
        return (std.catalog.get(fcall.getName().toString()) != null);
    }

    /**
    * Checks if the given function call may have side effects.
    * @param fcall the function call to be examined.
    * @return true if the function call has a side effect.
    */
    public static boolean isSideEffectFree(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        Set<Property> properties = std.catalog.get(fcall.getName().toString());
        return (!properties.contains(Property.SIDE_EFFECT_GLOBAL) &&
                !properties.contains(Property.SIDE_EFFECT_PARAMETER));
    }

    /**
    * Checks if the only side effect in the given function call is IO.
    * @param fcall the function call to be examined.
    * @return true if so.
    */
    public static boolean isSideEffectFreeExceptIO(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        Set<Property> properties = std.catalog.get(fcall.getName().toString());
        return (properties.contains(Property.SIDE_EFFECT_STDIO) ||
                properties.contains(Property.SIDE_EFFECT_FILEIO));
    }

    /** Constructs a new repository */
    private StandardLibrary() {
        catalog = new HashMap<String, Set<Property>>();
        addEntries();
    }

    /**
    * Adds each entry to the repository. The assigned properties are based on
    * the description of the standard library functions that should have a
    * function declarator (456 entries), following C99 (ISO/IEC 9899:1999).
    */
    private void addEntries() {
        ////////////////////////////////////////////////////////////////////////
        // START of C99
        ////////////////////////////////////////////////////////////////////////
        // assert.h
        add("assert", Property.MAY_NOT_RETURN);
        // complex.h - they are all pure functions.
        add("cabs");
        add("cabsf");
        add("cabsl");
        add("cacos");
        add("cacosf");
        add("cacosh");
        add("cacoshf");
        add("cacoshl");
        add("cacosl");
        add("carg");
        add("cargf");
        add("cargl");
        add("casin");
        add("casinf");
        add("casinh");
        add("casinhf");
        add("casinhl");
        add("casinl");
        add("catan");
        add("catanf");
        add("catanh");
        add("catanhf");
        add("catanhl");
        add("catanl");
        add("ccos");
        add("ccosf");
        add("ccosh");
        add("ccoshf");
        add("ccoshl");
        add("ccosl");
        add("cexp");
        add("cexpf");
        add("cexpl");
        add("cimag");
        add("cimagf");
        add("cimagl");
        add("clog");
        add("clogf");
        add("clogl");
        add("conj");
        add("conjf");
        add("conjl");
        add("cpow");
        add("cpowf");
        add("cpowl");
        add("cproj");
        add("cprojf");
        add("cprojl");
        add("creal");
        add("crealf");
        add("creall");
        add("csin");
        add("csinf");
        add("csinh");
        add("csinhf");
        add("csinhl");
        add("csinl");
        add("csqrt");
        add("csqrtf");
        add("csqrtl");
        add("ctan");
        add("ctanf");
        add("ctanh");
        add("ctanhf");
        add("ctanhl");
        add("ctanl");
        // ctype.h - they are all pure functions.
        add("isalnum");
        add("isalpha");
        add("isblank");
        add("iscntrl");
        add("isdigit");
        add("isgraph");
        add("islower");
        add("isprint");
        add("ispunct");
        add("isspace");
        add("isupper");
        add("isxdigit");
        add("tolower");
        add("toupper");
        // errno.h
        // fenv.h
        add("feclearexcept", Property.SIDE_EFFECT_GLOBAL);
        add("fegetexceptflag", Property.SIDE_EFFECT_PARAMETER);
        add("feraiseexcept", Property.SIDE_EFFECT_GLOBAL);
        add("fesetexceptflag", Property.SIDE_EFFECT_GLOBAL);
        add("fetestexcept");
        add("fegetround");
        add("fesetround", Property.SIDE_EFFECT_GLOBAL);
        add("fegetenv", Property.SIDE_EFFECT_PARAMETER);
        add("feholdexcept", Property.SIDE_EFFECT_PARAMETER);
        add("fesetenv", Property.SIDE_EFFECT_GLOBAL);
        add("feupdateenv", Property.SIDE_EFFECT_GLOBAL);
        // float.h
        // inttypes.h
        add("imaxabs");
        add("imaxdiv");
        add("strtoimax", Property.SIDE_EFFECT_PARAMETER);
        add("strtoumax", Property.SIDE_EFFECT_PARAMETER);
        add("wcstoimax", Property.SIDE_EFFECT_PARAMETER);
        add("wcstoumax", Property.SIDE_EFFECT_PARAMETER);
        // iso646.h
        // limits.h
        // locale.h - accesses environment variables.
        add("localeconv");
        add("setlocale", Property.SIDE_EFFECT_GLOBAL);
        // math.h - some functions return values through a pointer parameter.
        add("acos");
        add("acosf");
        add("acosh");
        add("acoshf");
        add("acoshl");
        add("acosl");
        add("asin");
        add("asinf");
        add("asinh");
        add("asinhf");
        add("asinhl");
        add("asinl");
        add("atan");
        add("atan2");
        add("atan2f");
        add("atan2l");
        add("atanf");
        add("atanh");
        add("atanhf");
        add("atanhl");
        add("atanl");
        add("cbrt");
        add("cbrtf");
        add("cbrtl");
        add("ceil");
        add("ceilf");
        add("ceill");
        add("copysign");
        add("copysignf");
        add("copysignl");
        add("cos");
        add("cosf");
        add("cosh");
        add("coshf");
        add("coshl");
        add("cosl");
        add("erf");
        add("erfc");
        add("erfcf");
        add("erfcl");
        add("erff");
        add("erfl");
        add("exp");
        add("exp2");
        add("exp2f");
        add("exp2l");
        add("expf");
        add("expl");
        add("expm1");
        add("expm1f");
        add("expm1l");
        add("fabs");
        add("fabsf");
        add("fabsl");
        add("fdim");
        add("fdimf");
        add("fdiml");
        add("floor");
        add("floorf");
        add("floorl");
        add("fma");
        add("fmaf");
        add("fmal");
        add("fmax");
        add("fmaxf");
        add("fmaxl");
        add("fmin");
        add("fminf");
        add("fminl");
        add("fmod");
        add("fmodf");
        add("fmodl");
        add("frexp", Property.SIDE_EFFECT_PARAMETER);
        add("frexpf", Property.SIDE_EFFECT_PARAMETER);
        add("frexpl", Property.SIDE_EFFECT_PARAMETER);
        add("hypot");
        add("hypotf");
        add("hypotl");
        add("ilogb");
        add("ilogbf");
        add("ilogbl");
        add("ldexp");
        add("ldexpf");
        add("ldexpl");
        add("lgamma");
        add("lgammaf");
        add("lgammal");
        add("llrint");
        add("llrintf");
        add("llrintl");
        add("llround");
        add("llroundf");
        add("llroundl");
        add("log");
        add("log10");
        add("log10f");
        add("log10l");
        add("log1p");
        add("log1pf");
        add("log1pl");
        add("log2");
        add("log2f");
        add("log2l");
        add("logb");
        add("logbf");
        add("logbl");
        add("logf");
        add("logl");
        add("lrint");
        add("lrintf");
        add("lrintl");
        add("lround");
        add("lroundf");
        add("lroundl");
        add("modf", Property.SIDE_EFFECT_PARAMETER);
        add("modff", Property.SIDE_EFFECT_PARAMETER);
        add("modfl", Property.SIDE_EFFECT_PARAMETER);
        add("nan");
        add("nanf");
        add("nanl");
        add("nearbyint");
        add("nearbyintf");
        add("nearbyintl");
        add("nextafter");
        add("nextafterf");
        add("nextafterl");
        add("nexttoward");
        add("nexttowardf");
        add("nexttowardl");
        add("pow");
        add("powf");
        add("powl");
        add("remainder");
        add("remainderf");
        add("remainderl");
        add("remquo", Property.SIDE_EFFECT_PARAMETER);
        add("remquof", Property.SIDE_EFFECT_PARAMETER);
        add("remquol", Property.SIDE_EFFECT_PARAMETER);
        add("rint");
        add("rintf");
        add("rintl");
        add("round");
        add("roundf");
        add("roundl");
        add("scalbln");
        add("scalblnf");
        add("scalblnl");
        add("scalbn");
        add("scalbnf");
        add("scalbnl");
        add("sin");
        add("sinf");
        add("sinh");
        add("sinhf");
        add("sinhl");
        add("sinl");
        add("sqrt");
        add("sqrtf");
        add("sqrtl");
        add("tan");
        add("tanf");
        add("tanh");
        add("tanhf");
        add("tanhl");
        add("tanl");
        add("tgamma");
        add("tgammaf");
        add("tgammal");
        add("trunc");
        add("truncf");
        add("truncl");
        // setjmp.h - accesses jmp_buf object.
        add("longjmp", Property.SIDE_EFFECT_PARAMETER);
        add("setjmp", Property.SIDE_EFFECT_PARAMETER);
        // signal.h - assume they change the execution environment.
        add("raise", Property.SIDE_EFFECT_GLOBAL);
        add("signal", Property.SIDE_EFFECT_GLOBAL);
        // stdarg.h - accesses va_list object.
        add("va_arg", Property.SIDE_EFFECT_PARAMETER);
        add("va_copy", Property.SIDE_EFFECT_PARAMETER);
        add("va_end", Property.SIDE_EFFECT_PARAMETER);
        add("va_start", Property.SIDE_EFFECT_PARAMETER);
        // stdbool.h
        // stddef.h
        // stdint.h
        // stdio.h
        add("clearerr", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fclose", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("feof");
        add("ferror");
        add("fflush", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fgetc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fgetpos", Property.SIDE_EFFECT_PARAMETER);
        add("fgets", Property.SIDE_EFFECT_PARAMETER);
        add("fopen");
        add("fprintf", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fputc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fputs", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fread", Property.SIDE_EFFECT_PARAMETER);
        add("freopen", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fscanf", Property.SIDE_EFFECT_PARAMETER);
        add("fseek", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fsetpos", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("ftell");
        add("fwrite", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("getc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("getchar", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("gets", Property.SIDE_EFFECT_PARAMETER);
        add("perror", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("printf", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("putc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("putchar", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("puts", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("remove", Property.SIDE_EFFECT_GLOBAL);     // file system
        add("rename", Property.SIDE_EFFECT_GLOBAL);     // file system
        add("rewind", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("scanf", Property.SIDE_EFFECT_PARAMETER);
        add("setbuf", Property.SIDE_EFFECT_PARAMETER);
        add("setvbuf", Property.SIDE_EFFECT_PARAMETER);
        add("snprintf", Property.SIDE_EFFECT_PARAMETER);
        add("sprintf", Property.SIDE_EFFECT_PARAMETER);
        add("sscanf", Property.SIDE_EFFECT_PARAMETER);
        add("tmpfile", Property.SIDE_EFFECT_GLOBAL);    //file system
        add("tmpnam", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_PARAMETER);
        add("ungetc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("vfprintf", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("vfscanf", Property.SIDE_EFFECT_PARAMETER);
        add("vprintf", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("vscanf", Property.SIDE_EFFECT_PARAMETER);
        add("vsnprintf", Property.SIDE_EFFECT_PARAMETER);
        add("vsprintf", Property.SIDE_EFFECT_PARAMETER);
        add("vsscanf", Property.SIDE_EFFECT_PARAMETER);
        // stdlib.h
        add("_Exit", Property.MAY_NOT_RETURN);
        add("abort", Property.MAY_NOT_RETURN);
        add("abs");
        add("atexit", Property.SIDE_EFFECT_GLOBAL);
        add("atof");
        add("atoi");
        add("atol");
        add("atoll");
        add("bsearch");
        add("calloc");
        add("div");
        add("exit", Property.MAY_NOT_RETURN);
        add("free", Property.SIDE_EFFECT_PARAMETER);
        add("getenv");
        add("labs");
        add("ldiv");
        add("llabs");
        add("lldiv");
        add("malloc");
        add("mblen");
        add("mbstowcs", Property.SIDE_EFFECT_PARAMETER);
        add("mbtowc", Property.SIDE_EFFECT_PARAMETER);
        add("qsort", Property.SIDE_EFFECT_PARAMETER);
        add("rand");
        add("realloc", Property.SIDE_EFFECT_PARAMETER);
        add("setenv", Property.SIDE_EFFECT_GLOBAL);
        add("srand", Property.SIDE_EFFECT_GLOBAL);
        add("strtod", Property.SIDE_EFFECT_PARAMETER);
        add("strtof", Property.SIDE_EFFECT_PARAMETER);
        add("strtol", Property.SIDE_EFFECT_PARAMETER);
        add("strtold", Property.SIDE_EFFECT_PARAMETER);
        add("strtoll", Property.SIDE_EFFECT_PARAMETER);
        add("strtoul", Property.SIDE_EFFECT_PARAMETER);
        add("strtoull", Property.SIDE_EFFECT_PARAMETER);
        add("system", Property.SIDE_EFFECT_GLOBAL);
        add("wcstombs", Property.SIDE_EFFECT_PARAMETER);
        add("wctomb", Property.SIDE_EFFECT_PARAMETER);
        // string.h
        add("memchr");
        add("memcmp");
        add("memcpy", Property.SIDE_EFFECT_PARAMETER);
        add("memmove", Property.SIDE_EFFECT_PARAMETER);
        add("memset", Property.SIDE_EFFECT_PARAMETER);
        add("strcat", Property.SIDE_EFFECT_PARAMETER);
        add("strchr");
        add("strcmp");
        add("strcoll");
        add("strcpy", Property.SIDE_EFFECT_PARAMETER);
        add("strcspn");
        add("strerror");
        add("strlen");
        add("strncat", Property.SIDE_EFFECT_PARAMETER);
        add("strncmp");
        add("strncpy", Property.SIDE_EFFECT_PARAMETER);
        add("strpbrk");
        add("strrchr");
        add("strspn");
        add("strstr");
        add("strtok", Property.SIDE_EFFECT_PARAMETER);
        add("strxfrm", Property.SIDE_EFFECT_PARAMETER);
        // tgmath.h - type-generic set of macros
        // time.h
        add("asctime");
        add("clock");
        add("ctime");
        add("difftime");
        add("gmtime");
        add("localtime");
        add("mktime", Property.SIDE_EFFECT_PARAMETER);
        add("strftime", Property.SIDE_EFFECT_PARAMETER);
        add("time", Property.SIDE_EFFECT_PARAMETER);
        // wchar.h
        add("btowc");
        add("fgetwc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fgetws", Property.SIDE_EFFECT_PARAMETER);
        add("fputwc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fputws", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fwide", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fwprintf", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("fwscanf", Property.SIDE_EFFECT_PARAMETER);
        add("getwc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("getwchar", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("mbrlen", Property.SIDE_EFFECT_PARAMETER);
        add("mbrtowc", Property.SIDE_EFFECT_PARAMETER);
        add("mbsinit");
        add("mbsrtowcs", Property.SIDE_EFFECT_PARAMETER);
        add("putwc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("putwchar", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("swprintf", Property.SIDE_EFFECT_PARAMETER);
        add("swscanf", Property.SIDE_EFFECT_PARAMETER);
        add("ungetwc", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("vfwprintf", Property.SIDE_EFFECT_PARAMETER,
                                    Property.SIDE_EFFECT_FILEIO);
        add("vfwscanf", Property.SIDE_EFFECT_PARAMETER);
        add("vwprintf", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("vswprintf", Property.SIDE_EFFECT_PARAMETER);
        add("vswscanf", Property.SIDE_EFFECT_PARAMETER);
        add("vwscanf", Property.SIDE_EFFECT_PARAMETER);
        add("wcrtomb", Property.SIDE_EFFECT_PARAMETER);
        add("wcscat", Property.SIDE_EFFECT_PARAMETER);
        add("wcschr");
        add("wcscmp");
        add("wcscoll");
        add("wcscpy", Property.SIDE_EFFECT_PARAMETER);
        add("wcscspn");
        add("wcsftime", Property.SIDE_EFFECT_PARAMETER);
        add("wcslen");
        add("wcsncat", Property.SIDE_EFFECT_PARAMETER);
        add("wcsncmp");
        add("wcsncpy", Property.SIDE_EFFECT_PARAMETER);
        add("wcspbrk");
        add("wcsrchr");
        add("wcsrtombs", Property.SIDE_EFFECT_PARAMETER);
        add("wcsspn");
        add("wcsstr");
        add("wcstod", Property.SIDE_EFFECT_PARAMETER);
        add("wcstof", Property.SIDE_EFFECT_PARAMETER);
        add("wcstok", Property.SIDE_EFFECT_PARAMETER);
        add("wcstol", Property.SIDE_EFFECT_PARAMETER);
        add("wcstold", Property.SIDE_EFFECT_PARAMETER);
        add("wcstoll", Property.SIDE_EFFECT_PARAMETER);
        add("wcstoul", Property.SIDE_EFFECT_PARAMETER);
        add("wcstoull", Property.SIDE_EFFECT_PARAMETER);
        add("wcsxfrm", Property.SIDE_EFFECT_PARAMETER);
        add("wctob");
        add("wmemchr");
        add("wmemcmp");
        add("wmemcpy", Property.SIDE_EFFECT_PARAMETER);
        add("wmemmove", Property.SIDE_EFFECT_PARAMETER);
        add("wmemset", Property.SIDE_EFFECT_PARAMETER);
        add("wprintf", Property.SIDE_EFFECT_GLOBAL,
                                    Property.SIDE_EFFECT_STDIO);
        add("wscanf", Property.SIDE_EFFECT_PARAMETER);
        // wctype.h
        add("iswalnum");
        add("iswalpha");
        add("iswblank");
        add("iswcntrl");
        add("iswdigit");
        add("iswgraph");
        add("iswlower");
        add("iswprint");
        add("iswpunct");
        add("iswspace");
        add("iswupper");
        add("iswxdigit");
        add("iswctype");
        add("towctrans");
        add("towlower");
        add("towupper");
        add("wctrans");
        add("wctype");
        // END of C99
    }

    /** Adds the specified properties to the call */
    private void add(String name, Property ... properties) {
        catalog.put(name, EnumSet.noneOf(Property.class));
        Set<Property> props = catalog.get(name);
        for (Property property : properties) {
            props.add(property);
        }
    }

    /* Automated categorizers */
    private static boolean takesVoid(ProcedureDeclarator pdecl) {
        return pdecl.getParameters().toString().equals("void ");
    }

    private static boolean returnsVoid(ProcedureDeclarator pdecl) {
        List types = pdecl.getTypeSpecifiers();
        return (types.size() == 1 && types.get(0) == Specifier.VOID);
    }

    private static int getNumMutableArguments(ProcedureDeclarator pdecl) {
        int ret = 0;
        if (takesVoid(pdecl)) {
            return ret;
        }
        for (Declaration decl : pdecl.getParameters()) {
            Symbol param = (Symbol)decl.getChildren().get(0);
            List types = param.getTypeSpecifiers();
            if (SymbolTools.isPointer(param) &&
                !types.contains(Specifier.CONST)) {
                ret++;
            }
        }
        return ret;
    }

    private static boolean containsVarArguments(ProcedureDeclarator pdecl) {
        return (pdecl.toString().contains("..."));
    }

    private static boolean containsStreamArguments(ProcedureDeclarator pdecl) {
        if (takesVoid(pdecl)) {
            return false;
        }
        for (Declaration decl : pdecl.getParameters()) {
            Symbol param = (Symbol)decl.getChildren().get(0);
            for (Object o : param.getTypeSpecifiers()) {
                if (o.toString().equals("FILE") ||
                    o.toString().equals("__FILE")) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean returnsPointer(ProcedureDeclarator pdecl) {
        return SymbolTools.isPointer(pdecl);
    }

    public static void addSideEffectParamIndices(String fname, int[]indices) {
        seIndices.put(fname, indices);
    }

    /**
    * Returns the position of the function call arguments that may have a side
    * effect upon a call.
    * @param fcall the function call to be inspected.
    * @return the list of positions having a side effect.
    */
    public static int[] getSideEffectParamIndices(FunctionCall fcall) {
        int num_args = fcall.getNumArguments();
        int[] ret = new int[num_args];
        if (fcall.getName() instanceof Identifier &&
            SymbolTools.getSymbolOf(fcall.getName()) != null) {
            Symbol symbol = ((Identifier)fcall.getName()).getSymbol();
            ProcedureDeclarator pdecl = null;
            if (symbol instanceof ProcedureDeclarator) {
                pdecl = (ProcedureDeclarator)symbol;
            } else if (symbol instanceof Procedure) {
                pdecl =(ProcedureDeclarator)((Procedure)symbol).getDeclarator();
            }
            if (pdecl != null) {
                List<Declaration> params = pdecl.getParameters();
                boolean prev_decision = true;
                int pos = 0;
                for (int i = 0; i < params.size(); i++) {
                    Declaration d = params.get(i);
                    Symbol p = (Symbol)d.getChildren().get(0);
                    // Handling of variable-length arguments
                    if (d.toString().trim().equals("...")) {
                        if (prev_decision) {
                            for (int j = i; j < num_args; j++) {
                                ret[pos++] = j;
                            }
                        }
                    } else {
                        prev_decision = false;
                        Specifier pu = PointerSpecifier.UNQUALIFIED;
                        Specifier cn = Specifier.CONST;
                        List specs = SymbolTools.getNativeSpecifiers(fcall, p);
                        if (specs == null ||
                            specs.contains(PointerSpecifier.CONST) ||
                            specs.contains(PointerSpecifier.CONST_VOLATILE) ||
                            specs.contains(PointerSpecifier.VOLATILE)) {
                            ret[pos++] = i;
                            prev_decision = true;
                        } else if (specs.contains(pu)) {
                            int pu0 = specs.indexOf(pu);
                            int pu1 = specs.lastIndexOf(pu);
                            int cn0 = specs.indexOf(cn);
                            int cn1 = specs.lastIndexOf(cn);
                            if (pu0 >= 0 && cn0 >= 0 &&
                                pu0 == pu1 && cn0 == cn1 &&
                                pu0 > cn0) {
                                // The only case having no side effect.
                            } else {
                                ret[pos++] = i;
                                prev_decision = true;
                            }
                        } else {
                            System.out.println(specs);
                            boolean has_array_spec = false;
                            for (int j = 0; j < specs.size(); j++) {
                                if (specs.get(j) instanceof ArraySpecifier) {
                                    has_array_spec = true;
                                    break;
                                }
                            }
                            if (has_array_spec && !specs.contains(cn)) {
                                ret[pos++] = i;
                                prev_decision = true;
                            }
                        }
                    }
                }
                return Arrays.copyOf(ret, pos);
            }
        }
        // returns conservative result for all other cases
        for (int i = 0; i < num_args; i++) {
            ret[i] = i;
        }
        return ret;
/*
        if (hasSideEffectOnParameter(fcall) == false) {
            return null;
        }
        return seIndices.get(fcall.getName().toString());
*/
    }

    public static boolean hasSideEffectOnParameter(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        Set<Property> properties = std.catalog.get(fcall.getName().toString());
        return properties.contains(Property.SIDE_EFFECT_PARAMETER) &&
                properties.contains(Property.SIDE_EFFECT_FILEIO) == false;
    }

}
