package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* <b>BinaryExpression</b> represents an expression having a LHS operand, an
* operator, and a RHS operand.
*/
public class BinaryExpression extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = BinaryExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** The binary operator of the expression */
    protected BinaryOperator op;

    /**
    * Creates a binary expression.
    *
    * @param lhs The lefthand expression.
    * @param op A binary operator.
    * @param rhs The righthand expression.
    * @throws NotAnOrphanException if <b>lhs</b> or <b>rhs</b> has a parent
    * object.
    */
    public BinaryExpression(
            Expression lhs, BinaryOperator op, Expression rhs) {
        super(2);
        object_print_method = class_print_method;
        addChild(lhs);
        this.op = op;
        addChild(rhs);
    }

    /** Returns a clone of the binary expression */
    @Override
    public BinaryExpression clone() {
        BinaryExpression o = (BinaryExpression)super.clone();
        o.op = op;
        return o;
    }

    /**
    * Prints a binary expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(BinaryExpression e, PrintWriter o) {
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
    * Returns a string representation of the binary expression. The returned
    * string is equivalent to the printed stream through {@link #defaultPrint}.
    * @return the string representation.
    */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        if (needs_parens) {
            sb.append("(");
        }
        sb.append(getLHS()).append(op).append(getRHS());
        if (needs_parens) {
            sb.append(")");
        }
        return sb.toString();
    }

    @Override
    protected int hashCode(int h) {
        if (needs_parens) {
            h = 31 * h + '(';
        }
        h = getLHS().hashCode(h);
        h = hashCode(op, h);
        h = getRHS().hashCode(h);
        if (needs_parens) {
            h = 31 * h + ')';
        }
        return h;
    }

    /**
    * Returns the lefthand expression.
    *
    * @return the lefthand expression.
    */
    public Expression getLHS() {
        return (Expression)children.get(0);
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
    * Returns the righthand expression.
    *
    * @return the righthand expression.
    */
    public Expression getRHS() {
        return (Expression)children.get(1);
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
    * Sets the lefthand expression.
    *
    * @param expr The new lefthand expression.  It must not be
    *   the child of another expression or null.
    * @throws IllegalArgumentException if <b>expr</b> is null.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void setLHS(Expression expr) {
        setChild(0, expr);
    }

    /**
    * Sets the operator for the expression.
    *
    * @param op The operator.
    */
    public void setOperator(BinaryOperator op) {
        this.op = op;
    }

    /**
    * Sets the righthand expression.
    *
    * @param expr The new righthand expression. It must not be the child of
    *       another expression.
    * @throws IllegalArgumentException if <b>expr</b> is null.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void setRHS(Expression expr) {
        setChild(1, expr);
    }

    /**
    * Compares the binary expression with the specified object.
    *
    * @return true if their childrens are equal and their operator is equal.
    */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) &&
                op.equals(((BinaryExpression)o).op));
    }

}
