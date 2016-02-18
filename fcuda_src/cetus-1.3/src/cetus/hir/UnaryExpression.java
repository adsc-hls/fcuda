package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents an expression having a unary operator and an operand expression.
*/
public class UnaryExpression extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = UnaryExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** The unary operator of the expression */
    protected UnaryOperator op;

    /**
    * Constructs a unary expression with the specified operator and expression.
    *
    * @param op the unary operator.
    * @param expr the operand expression.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    public UnaryExpression(UnaryOperator op, Expression expr) {
        object_print_method = class_print_method;
        this.op = op;
        addChild(expr);
    }

    /** Returns a clone of the unary expression. */
    @Override
    public UnaryExpression clone() {
        UnaryExpression o = (UnaryExpression)super.clone();
        o.op = op;
        return o;
    }

    /**
    * Prints a unary expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(UnaryExpression e, PrintWriter o) {
        if (e.needs_parens) {
            o.print("(");
        }
        if (e.op == UnaryOperator.POST_DECREMENT ||
            e.op == UnaryOperator.POST_INCREMENT) {
            e.getExpression().print(o);
            o.print(" ");
            e.op.print(o);
            o.print(" ");
        } else {
            o.print(" ");
            e.op.print(o);
            o.print(" ");
            e.getExpression().print(o);
        }
        if (e.needs_parens) {
            o.print(")");
        }
    }

    /**
    * Returns a string representation of the unary expression. The returned
    * string is equivalent to the printed stream through {@link #defaultPrint}.
    * @return the string representation.
    */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        if (needs_parens) {
            sb.append("(");
        }
        if (op == UnaryOperator.POST_DECREMENT ||
            op == UnaryOperator.POST_INCREMENT) {
            sb.append(getExpression());
            sb.append(" ");
            sb.append(op);
            sb.append(" ");
        } else {
            sb.append(" ");
            sb.append(op);
            sb.append(" ");
            sb.append(getExpression());
        }
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
        if (op == UnaryOperator.POST_DECREMENT ||
            op == UnaryOperator.POST_INCREMENT) {
            h = getExpression().hashCode(h);
            h = 31 * h + ' ';
            h = hashCode(op, h);
            h = 31 * h + ' ';
        } else {
            h = 31 * h + ' ';
            h = hashCode(op, h);
            h = 31 * h + ' ';
            h = getExpression().hashCode(h);
        }
        if (needs_parens) {
            h = 31 * h + ')';
        }
        return h;
    }

    /** Compares the unary expression with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && op == ((UnaryExpression)o).op);
    }

    /**
    * Returns the operand expression.
    *
    * @return the expression.
    */
    public Expression getExpression() {
        return (Expression)children.get(0);
    }

    /**
    * Returns the operator of the expression.
    *
    * @return the operator.
    */
    public UnaryOperator getOperator() {
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
    * Sets the operand expression with the specified new expression.
    *
    * @param expr the new operand expression.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    public void setExpression(Expression expr) {
        setChild(0, expr);
    }

}
