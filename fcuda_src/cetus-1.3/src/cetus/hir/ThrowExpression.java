package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** This class is no longer supported */
public class ThrowExpression extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ThrowExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    public ThrowExpression(Expression expr) {
        object_print_method = class_print_method;
        children.add(expr);
        expr.setParent(this);
    }

    @Override
    public ThrowExpression clone() {
        ThrowExpression o = (ThrowExpression)super.clone();
        return o;
    }

    /**
    * Prints a throw expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(ThrowExpression e, PrintWriter o) {
        o.print("throw ");
        e.getExpression().print(o);
    }

    public Expression getExpression() {
        return (Expression)children.get(0);
    }

}
