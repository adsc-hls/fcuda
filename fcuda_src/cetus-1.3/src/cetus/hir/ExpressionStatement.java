package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* A statement with the purpose of evaluating an expression for a side effect.
*/
public class ExpressionStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ExpressionStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create an expression statement given an expression.
    *
    * @param expr The expression part of the statement.
    * @throws IllegalArgumentException if <b>expr</b> is null.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    public ExpressionStatement(Expression expr) {
        object_print_method = class_print_method;
        addChild(expr);
        expr.setParens(false);
    }

    /**
    * Prints a statement to a stream.
    *
    * @param s The statement to print.
    * @param o The writer on which to print the statement.
    */
    public static void defaultPrint(ExpressionStatement s, PrintWriter o) {
        s.getExpression().print(o);
        o.print(";");
    }

    /**
    * Returns the expression part of the statement.
    *
    * @return the expression part of the statement.
    */
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

    /**
    * Clone this ExpressionStatement.
    * @return The cloned ExpressionStatement.
    */
    @Override public ExpressionStatement clone() {
        ExpressionStatement es = (ExpressionStatement)super.clone();
        return es;
    }

}
