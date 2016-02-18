package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents the only ternary expression in C and C++, the &#63;: operator.
*/
public class ConditionalExpression extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ConditionalExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Constructs a conditional expression with the specified condition, true
    * part, and false part.
    *
    * @param condition the condition expression.
    * @param true_expr the expression being evaluated if <b>condition</b> is
    *       true.
    * @param false_expr the expression being evaluated if <b>condition</b> is
    *       false.
    * @throws IllegalArgumentException if one of the parameters is invalid.
    * @throws NotAnOrphanException if one of the parameters has a parent object.
    */
    public ConditionalExpression(
            Expression condition, Expression true_expr, Expression false_expr) {
        super(3);
        object_print_method = class_print_method;
        addChild(condition);
        addChild(true_expr);
        addChild(false_expr);
    }

    /**
    * Prints a conditional expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(ConditionalExpression e, PrintWriter o) {
        o.print("(");
        e.getCondition().print(o);
        o.print(" ? ");
        e.getTrueExpression().print(o);
        o.print(" : ");
        e.getFalseExpression().print(o);
        o.print(")");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("(");
        sb.append(getCondition());
        sb.append(" ? ");
        sb.append(getTrueExpression());
        sb.append(" : ");
        sb.append(getFalseExpression());
        sb.append(")");
        return sb.toString();
    }

    @Override
    protected int hashCode(int h) {
        h = 31 * h + '(';
        h = getCondition().hashCode(h);
        h = hashCode(" ? ", h);
        h = getTrueExpression().hashCode(h);
        h = hashCode(" : ", h);
        h = getFalseExpression().hashCode(h);
        h = 31 * h + ')';
        return h;
    }

    /** Returns the condition expression. */
    public Expression getCondition() {
        return (Expression)children.get(0);
    }

    /** Returns the expression that follows the false jump. */
    public Expression getFalseExpression() {
        return (Expression)children.get(2);
    }

    /** Returns the expression that follows the true jump. */
    public Expression getTrueExpression() {
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
    * Sets the condition expression with the specifed new expression.
    *
    * @param expr the new condition expression.
    * @throws IllegalArgumentException if <b>expr</b> is invalid.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void setCondition(Expression expr) {
        setChild(0, expr);
    }

    /**
    * Sets the false expression with the specifed new expression.
    *
    * @param expr the new false expression.
    * @throws IllegalArgumentException if <b>expr</b> is invalid.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void setFalseExpression(Expression expr) {
        setChild(2, expr);
    }

    /**
    * Sets the true expression with the specifed new expression.
    *
    * @param expr the new true expression.
    * @throws IllegalArgumentException if <b>expr</b> is invalid.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void setTrueExpression(Expression expr) {
        setChild(1, expr);
    }

    @Override
    public ConditionalExpression clone() {
        ConditionalExpression ret = (ConditionalExpression)super.clone();
        return ret;
    }

}
