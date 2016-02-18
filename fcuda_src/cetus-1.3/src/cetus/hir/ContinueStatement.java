package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a continue statement: continue ';'
*/
public class ContinueStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ContinueStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create a new continue statement.
    */
    public ContinueStatement() {
        super(0);
        object_print_method = class_print_method;
    }

    /**
    * Prints a continue statement to a stream.
    *
    * @param s The statement to print.
    * @param o The writer on which to print the statement.
    */
    public static void defaultPrint(ContinueStatement s, PrintWriter o) {
        o.print("continue;");
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

    /** Returns a clone of the continue statement. */
    @Override
    public ContinueStatement clone() {
        ContinueStatement cs = (ContinueStatement)super.clone();
        return cs;
    }

}
