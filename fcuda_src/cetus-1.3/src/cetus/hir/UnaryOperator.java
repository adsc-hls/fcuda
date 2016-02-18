package cetus.hir;

import java.io.PrintWriter;
import java.util.HashMap;

/**
* Operators that act on a single expression.
*/
public class UnaryOperator implements Printable {

    private static HashMap<String, UnaryOperator> op_map =
            new HashMap<String, UnaryOperator>(16);

    private static String[] names = {
            "&", "~", "*", "!", "-", "+", "--", "++", "--", "++" };

    /**
    * &amp;
    */
    public static final UnaryOperator ADDRESS_OF = new UnaryOperator(0);

    /**
    * ~
    */
    public static final UnaryOperator BITWISE_COMPLEMENT = new UnaryOperator(1);

    /**
    * *
    */
    public static final UnaryOperator DEREFERENCE = new UnaryOperator(2);

    /**
    * !
    */
    public static final UnaryOperator LOGICAL_NEGATION = new UnaryOperator(3);

    /**
    * -
    */
    public static final UnaryOperator MINUS = new UnaryOperator(4);

    /**
    * +
    */
    public static final UnaryOperator PLUS = new UnaryOperator(5);

    /**
    * --
    */
    public static final UnaryOperator POST_DECREMENT = new UnaryOperator(6);

    /**
    * ++
    */
    public static final UnaryOperator POST_INCREMENT = new UnaryOperator(7);

    /**
    * --
    */
    public static final UnaryOperator PRE_DECREMENT = new UnaryOperator(8);

    /**
    * ++
    */
    public static final UnaryOperator PRE_INCREMENT = new UnaryOperator(9);

    protected int value;

    /**
    * Used internally -- you may not create arbitrary unary operators
    * and may only use the ones provided as static members.
    *
    * @param value The numeric code of the operator.
    */
    private UnaryOperator(int value) {
        this.value = value;
        op_map.put(names[value], this);
    }

    /** Creates a unary operator from the specified string. */
    public static UnaryOperator fromString(String s) {
        return op_map.get(s);
    }

    /** Checks if the specified operator implies a side effect. */
    public static boolean hasSideEffects(UnaryOperator op) {
        return (op == UnaryOperator.POST_DECREMENT ||
                op == UnaryOperator.POST_INCREMENT ||
                op == UnaryOperator.PRE_DECREMENT ||
                op == UnaryOperator.PRE_INCREMENT);
    }

    /* It is not necessary to override equals or provide cloning, because
       all possible operators are provided as static objects. */

    /** Prints the operator on the specifier print writer. */
    public void print(PrintWriter o) {
        o.print(names[value]);
    }

    /** Returns a string representation of the operator. */
    @Override
    public String toString() {
        return names[value];
    }

    /**
    * Verifies this operator is valid.
    *
    * @throws IllegalStateException if the operator is invalid.
    */
    public void verify() throws IllegalStateException {
        if (value < 0 || value > 9) {
            throw new IllegalStateException();
        }
    }

}
