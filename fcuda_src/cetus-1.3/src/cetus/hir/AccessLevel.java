package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
 * Provides a restricted set of specifiers for access levels.
 * In C++, a class member's access level is determined by the nearest
 * preceding access level specifier; an access level may apply to
 * numerous subsequent class members.  In Java (which may eventually be
 * supported by Cetus) access levels appear in many of the same
 * places as type specifiers do.  Thus, in this IR, AccessLevel
 * implements the C++ behavior, while specifiers of the same name
 * in {@link Specifier Specifier} implement the Java behavior.
 * This class is not used in C programs.
 */
@Deprecated
public final class AccessLevel extends Declaration {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class <?>[2];

        try {
            params[0] = AccessLevel.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private Specifier spec;

    /**
    * This constructor is used only within this class.
    */
    public AccessLevel(Specifier spec) {
        object_print_method = class_print_method;
        children = null;
        this.spec = spec;
    }

    /**
    * Prints an access level to a stream.
    *
    * @param al The access level to print.
    * @param o The writer on which to print the access level.
    */
    public static void defaultPrint(AccessLevel al, PrintWriter o) {
        al.spec.print(o);
        o.println(":");
    }

    /* It is not necessary to override equals or provide cloning, because
       all possible operators are provided as static objects. */
    @SuppressWarnings("unchecked")
    public List<IDExpression> getDeclaredIDs() {
        return (List<IDExpression>)empty_list;
    }

    /**
    * Overrides the class print method, so that all subsequently
    * created objects will use the supplied method.
    *
    * @param m The new print method.
    */
    static public void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /**
    * Unsupported - this object has no children.
    */
    public void setChild(int index, Traversable t) {
        throw new UnsupportedOperationException();
    }

    public void setParent(Traversable t) {
        if (t == null) {
            parent = null;
        } else {
            // parent must already have accepted this object as a child
            if (Tools.identityIndexOf(t.getChildren(), this) < 0) {
                throw new NotAChildException();
            }
            if (t instanceof ClassDeclaration) {
                parent = t;
            } else {
                throw new IllegalArgumentException();
            }
        }
    }

}
