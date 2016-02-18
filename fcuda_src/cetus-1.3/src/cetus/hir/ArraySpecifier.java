package cetus.hir;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
* Represents an array specifier, for example the bracketed
* parts of <var>int array[20][30];</var>
*/
public class ArraySpecifier extends Specifier {

    /** The unbounded specifier [] */
    public static final ArraySpecifier UNBOUNDED = new ArraySpecifier();

    private List<Expression> dimensions;

    /**
    * Constructs a new array specifier with an empty dimension.
    */
    public ArraySpecifier() {
        dimensions = new ArrayList<Expression>(1);
        dimensions.add(null);
    }
    
    /**
    * Constructs a new array specifier with the given dimension expression.
    * @param expr the dimension expression.
    */
    public ArraySpecifier(Expression expr) {
        dimensions = new ArrayList<Expression>(1);
        dimensions.add(expr);
    }

    /**
    * Constructs a new array specifier with the given dimension expressions.
    * @param dimensions the list of dimension expressions.
    */
    public ArraySpecifier(List dimensions) {
        setDimensions(dimensions);
    }

    /**
    * Gets the nth dimension of this array specifier.
    *
    * @param n The position of the dimension.
    * @throws IndexOutOfBoundsException if there is no expression at that
    *       position.
    * @return the nth dimension, which may be null.  A null dimension occurs
    *       for example with int array[][8].
    */
    public Expression getDimension(int n) {
        return dimensions.get(n);
    }

    /**
    * Returns the number of index expressions used in this array specifier.
    *
    * @return the number of index expressions.
    */
    public int getNumDimensions() {
        return dimensions.size();
    }

    public void print(PrintWriter o) {
        for (int i = 0; i < dimensions.size(); i++) {
            Expression dim = dimensions.get(i);
            o.print("[");
            if (dim != null) {
                dim.print(o);
            }
            o.print("]");
        }
    }

    /**
    * Sets the nth dimension of this array specifier.
    *
    * @param n The position of the dimension.
    * @param expr The expression defining the size of the dimension.
    * @throws IndexOutOfBoundsException if there is no dimension at that
    *       position.
    */
    public void setDimension(int n, Expression expr) {
        dimensions.set(n, expr);
    }

    /**
    * Set the list of dimension expressions.
    *
    * @param dimensions A list of expressions.
    */
    public void setDimensions(List dimensions) {
        if (dimensions == null) {
            throw new IllegalArgumentException();
        }
        this.dimensions = new ArrayList<Expression>(dimensions.size());
        for (int i = 0; i < dimensions.size(); i++) {
            Expression e = (Expression)dimensions.get(i);
            if (e != null && !(e instanceof Expression)) {
                throw new IllegalArgumentException(
                        "all list items must be Expressions or null; found a " +
                        e.getClass().getName() + " instead");
            }
            this.dimensions.add(e);
        }
    }

}
