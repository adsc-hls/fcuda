package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a case for use with switch.
*/
public class Case extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Case.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create a new case with the specified expression.
    *
    * @param expr The expression that activates the case.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    public Case(Expression expr) {
        object_print_method = class_print_method;
        addChild(expr);
    }

    /**
    * Prints a case label to a stream.
    *
    * @param c The case label to print.
    * @param o The writer on which to print the case label.
    */
    public static void defaultPrint(Case c, PrintWriter o) {
        o.print("case ");
        c.getExpression().print(o);
        o.print(":");
    }

    /** Returns the target expression associated with the case statement. */
    public Expression getExpression() {
        return (Expression)children.get(0);
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
