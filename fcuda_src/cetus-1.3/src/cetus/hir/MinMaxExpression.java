package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.Set;
import java.util.TreeSet;

/**
* Class for MIN() MAX() expressions - these are equivalent to conditional
* expressions.
*/
public class MinMaxExpression extends Expression {

    private static Method class_print_method;

    // MIN:true, MAX:false
    private boolean ismin;

    // Default print method
    static {
        try {
            class_print_method = MinMaxExpression.class.getMethod(
                    "defaultPrint", new Class<?>[] {MinMaxExpression.class,
                                                    PrintWriter.class});
        } catch(NoSuchMethodException ex) {
            throw new InternalError();
        }
    }

    /**
    * Prints out MIN/MAX expressions on the specified print writer.
    */
    public static void defaultPrint(MinMaxExpression m, PrintWriter o) {
        o.print((m.ismin) ? "MIN(" : "MAX(");
        PrintTools.printListWithComma(m.children, o);
        o.print(")");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        if (ismin) {
            sb.append("MIN(");
        } else {
            sb.append("MAX(");
        }
        sb.append(PrintTools.listToString(children, ", "));
        sb.append(")");
        return sb.toString();
    }

    @Override
    public int hashCode(int h) {
        if (ismin) {
            h = hashCode("MIN(", h);
        } else {
            h = hashCode("MAX(", h);
        }
        h = hashCode(children, ", ", h);
        h = 31 * h + ')';
        return h;
    }

    /**
    * Constructs a min/max expression with the specified operator description.
    *
    * @param ismin true for MIN, false for MAX.
    */
    public MinMaxExpression(boolean ismin) {
        super(2);
        object_print_method = class_print_method;
        this.ismin = ismin;
    }

    /**
    * Consructs a min/max expression with the specified operator flag and the
    * two operand expressions.
    * @param ismin true if the operator is MIN, false otherwise
    * @param lhs   the first operand
    * @param rhs   the second operand
    * @throws IllegalArgumentException if <b>lhs</b> or <b>rhs</b> is invalid.
    * @throws NotAnOrphanException if <b>lhs</b> or <b>rhs</b> has a parent.
    */
    public MinMaxExpression(boolean ismin, Expression lhs, Expression rhs) {
        this(ismin);
        addChild(lhs);
        addChild(rhs);
    }

    /**
    * Returns true if this is a MIN expression.
    * @return true if this is a MIN expression, false otherwise
    */
    public boolean isMin() {
        return ismin;
    }

    /**
    * Sets the operator with the specified flag.
    * @param ismin true for MIN, false for MAX
    */
    public void setMin(boolean ismin) {
        this.ismin = ismin;
    }

    /**
    * Adds an operand at the end of the operand list.
    * @param e the new operand to be appended.
    * @throws IllegalArgumentException if <b>e</b> is invalid.
    * @throws NotAnOrphanException if <b>e</b> has a parent.
    */
    public void add(Expression e) {
        addChild(e);
    }

    /**
    * Negates the expression by switching the operator and the signs of the
    * operands.
    * @return the negated expression
    */
    public Expression negate() {
        MinMaxExpression ret = new MinMaxExpression(!ismin);
        for (int i = 0; i < children.size(); i++) {
            Expression child = (Expression)children.get(i);
            ret.add(Symbolic.subtract(new IntegerLiteral(0), child));
        }
        return ret;
    }

    /**
    * Checks if this expression is a MIN expression having at least one operand
    * with positive value.
    * @return true/false
    */
    public boolean isPosMax() {
        if (ismin) {
            return false;
        }
        for (int i = 0; i < children.size(); i++) {
            Traversable child = children.get(i);
            if (child instanceof IntegerLiteral &&
                ((IntegerLiteral)child).getValue() > 0) {
                return true;
            }
        }
        return false;
    }

    /**
    * Checks if this expression is a MIN expression having at least one operand
    * with negative value.
    * @return true/false
    */
    public boolean isNegMin() {
        if (!ismin) {
            return false;
        }
        for (int i = 0; i < children.size(); i++) {
            Traversable child = children.get(i);
            if (child instanceof IntegerLiteral &&
                ((IntegerLiteral)child).getValue() < 0) {
                return true;
            }
        }
        return false;
    }

    // Return a non-minmax expression if comparison is handy
    // Only IntegerLiteral and Inf are captured
    private Expression resolve() {
        long prev = 0, curr = 0;
        for (int i = 0; i < children.size(); i++) {
            Traversable o = children.get(i);
            if (!(o instanceof IntegerLiteral) && !(o instanceof InfExpression))
                return this;
            if (o instanceof IntegerLiteral) {
                curr = ((IntegerLiteral)o).getValue();
            } else {
                curr = (((InfExpression)o).sign() < 0) ?
                        Long.MIN_VALUE : Long.MAX_VALUE;
            }
            if (o == children.get(0) || curr < prev && ismin ||
                curr > prev && !ismin) {
                prev = curr;
            }
        }
        if (prev == Long.MIN_VALUE || prev == Long.MAX_VALUE) {
            return new InfExpression((prev < 0) ? -1 : 1);
        } else {
            return new IntegerLiteral(prev);
        }
    }

    /**
    * Simplifies a MinMaxExpression as much as possible by removing duplicates
    * and by matching special cases.
    * @return A simplified expression
    */
    // Remove duplicates and matches MIN(MAX(a,b),a)
    public Expression simplify() {
        Set<Traversable> unique_children = new TreeSet<Traversable>();
        unique_children.addAll(children);
        // Only one unique child
        if (unique_children.size() == 1) {
            return (Expression)children.get(0);
        }
        MinMaxExpression ret = new MinMaxExpression(ismin);
        for (Traversable child : unique_children) {
            ret.add((Expression)child);
        }
        if (ret.children.size() == 2) {
            // Match simplifiable cases
            Expression e1 = (Expression)ret.children.get(0);
            Expression e2 = (Expression)ret.children.get(1);
            if (e1 instanceof MinMaxExpression &&
                e1.getChildren().size() == 2 &&
                ((MinMaxExpression) e1).isMin() != ret.isMin() &&
                (e2.equals(e1.getChildren().get(0)) ||
                        e2.equals(e1.getChildren().get(1)))) {
                return e2;
            } else if (e2 instanceof MinMaxExpression &&
                       e2.getChildren().size() == 2 &&
                       ((MinMaxExpression) e2).isMin() != ret.isMin() &&
                       (e1.equals(e2.getChildren().get(0)) ||
                                e1.equals(e2.getChildren().get(1)))) {
                return e1;
            }
        }
        return ret.resolve();
    }

    /**
    * Compares the min/max expression to the specified expression for order.
    */
    public int compareTo(Expression e) {
        if (e instanceof MinMaxExpression) {
            if (ismin == ((MinMaxExpression)e).ismin) {
                return super.compareTo(e);
            } else {
                return -1;
            }
        } else {
            return -1;
        }
    }

    /**
    * Compares MinMaxExpression to an integer value.
    * @param num an integer value
    * @return Integer object if comparison is possible, null otherwise
    */
    public Integer compareTo(int num) {
        for (int i = 0; i < children.size(); i++) {
            Traversable o = children.get(i);
            if (o instanceof IntegerLiteral) {
                if (ismin && ((IntegerLiteral)o).getValue() < num) {
                    return new Integer(-1);
                } else if (!ismin && ((IntegerLiteral) o).getValue() > num) {
                    return new Integer(1);
                }
            }
        }
        return null;
    }

    /**
    * Converts MinMaxExpression to an equivalent conditional expression.
    * @return the converted conditional expression
    */
    public Expression toConditionalExpression() {
        Expression ret = ((Expression)children.get(0)).clone();
        BinaryOperator rel = ismin ?
                BinaryOperator.COMPARE_LE : BinaryOperator.COMPARE_GE;
        for (int i = 1; i < children.size(); ++i) {
            ret = new ConditionalExpression(
                    new BinaryExpression(
                            ret.clone(),
                            (ismin ? BinaryOperator.COMPARE_LE :
                                     BinaryOperator.COMPARE_GE),
                            ((Expression)children.get(i)).clone()),
                    ret.clone(),
                    ((Expression)children.get(i)).clone());
        }
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && ismin == ((MinMaxExpression)o).ismin);
    }

}
