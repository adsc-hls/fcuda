package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a default for use with switch.
*/
public class Default extends Statement {

    private static Method class_print_method;
    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Default.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create a new default.
    */
    public Default() {
        super(-1);
        object_print_method = class_print_method;
    }

    /**
    * Prints a default label to a stream.
    *
    * @param d The default label to print.
    * @param o The writer on which to print the default label.
    */
    public static void defaultPrint(Default d, PrintWriter o) {
        o.print("default:");
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
