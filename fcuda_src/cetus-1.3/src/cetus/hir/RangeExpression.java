package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* RangeExpression represents a symbolic range with a lower bound
* expression and an upper bound expression. RangeAnalysis uses RangeExpression
* to compute a valid value ranges of variables at program points.
*/
public class RangeExpression extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = RangeExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch (NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Constructs a range expression with the given lower and upper bounds.
    *
    * @param lb the lower bound expression.
    * @param ub the upper bound expression.
    * @throws IllegalArgumentException if <b>lb</b> or <b>ub</b> is invalid.
    * @throws NotAnOrphanException if <b>lb</b> or <b>ub</b> has a parent.
    */
    public RangeExpression(Expression lb, Expression ub) {
      super(2);
      object_print_method = class_print_method;
      addChild(lb);
      addChild(ub);
    }

    /**
    * Prints the range expression on the specified print writer.
    */
    public static void defaultPrint(RangeExpression e, PrintWriter o) {
        o.print("[");
        e.getLB().print(o);
        o.print(":");
        e.getUB().print(o);
        o.print("]");
    }

    /**
    * Returns a string representation of the range expression. The returned
    * string is equivalent to the printed stream through {@link #defaultPrint}.
    * @return the string representation.
    */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("[");
        sb.append(getLB());
        sb.append(":");
        sb.append(getUB());
        sb.append("]");
        return sb.toString();
    }

    @Override
    protected int hashCode(int h) {
        h = 31*h + '[';
        h = getLB().hashCode(h);
        h = 31*h + ':';
        h = getUB().hashCode(h);
        h = 31*h + ']';
        return h;
    }

    /**
    * Compares the range expression with the specified expression for order.
    * @return 0 if they are equal, -1 otherwise.
    */
    public int compareTo(Expression e) {
        return (equals(e))? 0: -1;
    }

    /**
    * Sets the lower bound with the specified expression.
    * @param lb the new lower bound.
    * @throws NotAnOrphanException if <b>lb</b> has a parent.
    */
    public void setLB(Expression lb) {
        setChild(0, lb);
    }

    /**
    * Sets the upper bound with the specified expression.
    * @param ub the new upper bound
    * @throws NotAnOrphanException if <b>ub</b> has a parent.
    */
    public void setUB(Expression ub) {
        setChild(1, ub);
    }

    /**
    * Returns the lower bound of this range expression.
    */
    public Expression getLB() {
        return (Expression)children.get(0);
    }

    /**
    * Returns the upper bound of this range expression.
    */
    public Expression getUB() {
        return (Expression)children.get(1);
    }

    /**
    * Checks if this range expression has a lower bound greater than the upper
    * bound. The comparison is possible only for literals and infinity
    * expression.
    */
    public boolean isEmpty() {
        Expression lb = getLB(), ub = getUB();
        return (lb instanceof InfExpression && ((InfExpression)lb).sign() > 0 ||
                ub instanceof InfExpression && ((InfExpression)ub).sign() < 0 ||
                lb instanceof IntegerLiteral && ub instanceof IntegerLiteral &&
                ((IntegerLiteral)lb).getValue() >
                                ((IntegerLiteral)ub).getValue());
    }

    /**
    * Checks if this range expression does not have any closed bounds.
    */
    public boolean isOmega() {
        Expression lb = getLB(), ub = getUB();
        return (lb instanceof InfExpression && ((InfExpression)lb).sign() < 0 &&
                ub instanceof InfExpression && ((InfExpression)ub).sign() > 0);
    }

    /**
    * Returns true if neither lb nor ub is infinity.
    */
    public boolean isBounded() {
        Expression lb = getLB(), ub = getUB();
        return (!(lb instanceof InfExpression) &&
                !(ub instanceof InfExpression));
    }

    /**
    * Converts an arbitrary expression to a range expression by setting the same
    * lower bound and upper bound.
    */
    public static RangeExpression toRange(Expression e) {
        if (e instanceof RangeExpression) {
            return (RangeExpression)e.clone();
        }
        return new RangeExpression(e.clone(), e.clone());
    }

    /**
    * Converts this range expression to a non-range expression if the lower
    * bound is equal to the upper bound.
    */
    public Expression toExpression() {
        if (getLB().equals(getUB())) {
            return getLB();
        } else {
            return this;
        }
    }

    /**
    * Returns a new instanceof omega expression which is [-inf:inf].
    * @return a new omega expression.
    */
    public static Expression getOmega() {
        return new RangeExpression(new InfExpression(-1), new InfExpression(1));
    }

}
