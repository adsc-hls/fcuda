package cetus.hir;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
* Infix operators that act on two expressions.
*/
public class BinaryOperator implements Printable {

    private static HashMap<String, BinaryOperator> op_map =
            new HashMap<String, BinaryOperator>(32);

    private static String[] names = {
            "+", "&", "^", "|", "==", ">=", ">", "<=", "<",
            "!=", "/", "&&", "||", "%", "*", "<<", ">>", "-", " instanceof "};

    /**
    * +
    */
    public static final BinaryOperator ADD = new BinaryOperator(0);

    /**
    * &amp;
    */
    public static final BinaryOperator BITWISE_AND = new BinaryOperator(1);

    /**
    * ^
    */
    public static final BinaryOperator BITWISE_EXCLUSIVE_OR =
            new BinaryOperator(2);

    /**
    * |
    */
    public static final BinaryOperator BITWISE_INCLUSIVE_OR =
            new BinaryOperator(3);

    /**
    * ==
    */
    public static final BinaryOperator COMPARE_EQ = new BinaryOperator(4);

    /**
    * &gt;=
    */
    public static final BinaryOperator COMPARE_GE = new BinaryOperator(5);

    /**
    * &gt;
    */
    public static final BinaryOperator COMPARE_GT = new BinaryOperator(6);

    /**
    * &lt;=
    */
    public static final BinaryOperator COMPARE_LE = new BinaryOperator(7);

    /**
    * &lt;
    */
    public static final BinaryOperator COMPARE_LT = new BinaryOperator(8);

    /**
    * &#33;=
    */
    public static final BinaryOperator COMPARE_NE = new BinaryOperator(9);

    /**
    * /
    */
    public static final BinaryOperator DIVIDE = new BinaryOperator(10);

    /**
    * &amp;&amp;
    */
    public static final BinaryOperator LOGICAL_AND = new BinaryOperator(11);

    /**
    * ||
    */
    public static final BinaryOperator LOGICAL_OR = new BinaryOperator(12);

    /**
    * %
    */
    public static final BinaryOperator MODULUS = new BinaryOperator(13);

    /**
    * *
    */
    public static final BinaryOperator MULTIPLY = new BinaryOperator(14);

    /**
    * &lt;&lt;
    */
    public static final BinaryOperator SHIFT_LEFT = new BinaryOperator(15);

    /**
    * &gt;&gt;
    */
    public static final BinaryOperator SHIFT_RIGHT = new BinaryOperator(16);

    /** 
    * -
    */
    public static final BinaryOperator SUBTRACT = new BinaryOperator(17);

    /**
    * instanceof
    */
    public static final BinaryOperator INSTANCEOF = new BinaryOperator(18);

    protected int value;

    protected BinaryOperator() {
    }

    /**
    * Used internally -- you may not create arbitrary binary operators
    * and may only use the ones provided as static members.
    *
    * @param value The numeric code of the operator.
    */
    private BinaryOperator(int value) {
        this.value = value;
        op_map.put(names[value], this);
    }

    /**
    * Returns a binary operator that matches the specified string <tt>s</tt>.
    * @param s the string to be matched.
    * @return the matching operator or null if not found.
    */
    public static BinaryOperator fromString(String s) {
        return op_map.get(s);
    }

    /* It is not necessary to override equals or provide cloning, because
       all possible operators are provided as static objects. */

    public void print(PrintWriter o) {
        o.print(names[value]);
    }

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
        if (value < 0 || value > 18) {
            throw new IllegalStateException();
        }
    }
    
    /**
    * Checks if this operator belongs to binary comparison operator.
    */
    public boolean isCompare() {
        return (value >= 4 && value <= 9);
    }

    /**
    * Checks if this operator belongs to boolean logic operator.
    */
    public boolean isLogical() {
        return (value >= 11 && value <= 12);
    }

}
