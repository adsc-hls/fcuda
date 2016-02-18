package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* <b>InfExpression</b> represents a literal expression having a value greater
* than or less than any possible numbers. This class was introduced to express
* value range of an integer-typed variables.
*/
public class InfExpression extends Expression {

    private static Method class_print_method;

    // Sign of the expression
    private int sign;

    static {
        try {
            class_print_method = InfExpression.class.getMethod("defaultPrint",
                    new Class<?>[] {InfExpression.class, PrintWriter.class});
        } catch(NoSuchMethodException ex) {
            throw new InternalError();
        }
    }

    /**
    * Constructs an inf expression with the specified sign.
    *
    * @param sign the sign of the infinity expression
    */
    public InfExpression(int sign) {
        super(-1);
        object_print_method = class_print_method;
        this.sign = sign;
    }

    /**
    * Returns a clone of the inf expression.
    */
    @Override
    public InfExpression clone() {
        InfExpression o = (InfExpression)super.clone();
        o.sign = sign;
        return o;
    }

    /**
    * Prints the expression to the specified target.
    * @param e the expression to print.
    * @param o the writer on which to print the expression.
    */
    public static void defaultPrint(InfExpression e, PrintWriter o) {
        if (e.sign < 0) {
            o.print("-INF");
        } else {
            o.print("+INF");
        }
    }

    @Override
    public String toString() {
        return (sign < 0) ? "-INF" : "+INF";
    }

    /**
    * Compares the expression with the specified object.
    */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) &&
                (sign > 0) == (((InfExpression)o).sign > 0));
    }

    // InfExpression is not comparable to any objects
    public int compareTo(Expression e) {
        if (equals(e)) {
            return 0;
        } else {
            return toString().compareTo(e.toString());
        }
    }

    /**
    * Returns the sign of the infinity expression.
    */
    public int sign() {
        return sign;
    }

}
