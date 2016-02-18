package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.List;
import java.util.ArrayList;

/** This class is not supported */
public class NamespaceAlias extends Declaration {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = NamespaceAlias.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    private IDExpression name;

    private IDExpression original;

    /**
    * Create a namespace alias.
    *
    * @param name The name for this alias.
    * @param original The name of the original namespace.
    */
    public NamespaceAlias(IDExpression name, IDExpression original) {
        this.name = name;
        this.original = original;
    }

    /**
    * Prints an alias to a stream.
    *
    * @param a The alias to print.
    * @param o The writer on which to print the alias.
    */
    public static void defaultPrint(NamespaceAlias a, PrintWriter o) {
        o.print("namespace ");
        a.name.print(o);
        o.print(" = ");
        a.original.print(o);
    }

    public List<IDExpression> getDeclaredIDs() {
        List<IDExpression> ret = new ArrayList<IDExpression>(1);
        ret.add(name);
        return ret;
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

}
