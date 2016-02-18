package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* <b>AccessExpression</b> is a representation for expressions that access
* a member of class-like objects.
*/
public class AccessExpression extends BinaryExpression {

    /** Class print method by default */
    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = AccessExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Creates an assignment expression.
    *
    * @param lhs The lefthand expression.
    * @param op An assignment operator.
    * @param rhs The righthand expression.
    * @throws NotAnOrphanException if <b>lhs</b> or <b>rhs</b> has a parent
    * object.
    */
    public AccessExpression(Expression lhs, AccessOperator op, Expression rhs) {
        super(lhs, op, rhs);
        setParens(false);
        rhs.setParens(false);
    }

    /**
    * Returns a cloned <b>AccessExpression</b> object by calling the super
    * class, {@link BinaryExpression}'s clone method.
    *
    * @return the cloned <b>AccessExpression</b>.
    */
    @Override
    public AccessExpression clone() {
        return (AccessExpression) super.clone();
    }

    /**
    * Prints an assignment expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(AccessExpression e, PrintWriter o) {
        if (e.needs_parens) {
            o.print("(");
        }
        e.getLHS().print(o);
        e.op.print(o);
        e.getRHS().print(o);
        if (e.needs_parens) {
            o.print(")");
        }
    }

    /**
    * Returns the operator of the expression.
    *
    * @return the operator.
    */
    public BinaryOperator getOperator() {
        return op;
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
    * Sets the operator for the expression.
    *
    * @param op The operator.
    */
    public void setOperator(AccessOperator op) {
        this.op = op;
    }

}
