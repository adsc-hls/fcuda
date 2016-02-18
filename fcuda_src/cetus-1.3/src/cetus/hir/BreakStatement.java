package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a break statement: break ';'
*/
public class BreakStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = BreakStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create a new break statement.
    */
    public BreakStatement() {
        super(-1);
        object_print_method = class_print_method;
    }

    /**
    * Prints a break statement to a stream.
    *
    * @param s The statement to print.
    * @param o The writer on which to print the statement.
    */
    public static void defaultPrint(BreakStatement s, PrintWriter o) {
        o.print("break;");
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

    /** Returns a clone of the break statement. */
    @Override
    public BreakStatement clone() {
        BreakStatement bs = (BreakStatement)super.clone();
        return bs;
    }

}
