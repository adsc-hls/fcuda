package cetus.hir;

import java.io.PrintWriter;
import java.util.HashMap;

/**
* Infix operators that assign the value of their righthand side to their
* lefthand side.
*/
public class AssignmentOperator extends BinaryOperator {

    private static HashMap<String, AssignmentOperator> op_map =
            new HashMap<String, AssignmentOperator>(24);

    private static String[] names = {
            "+=", "&=", "^=", "|=", "/=", "=", "%=", "*=", "<<=", ">>=", "-="};

    /**
    * +=
    */
    public static final AssignmentOperator ADD = new AssignmentOperator(0);

    /**
    * &amp;=
    */
    public static final AssignmentOperator BITWISE_AND =
            new AssignmentOperator(1);

    /**
    * ^=
    */
    public static final AssignmentOperator BITWISE_EXCLUSIVE_OR =
            new AssignmentOperator(2);

    /**
    * |=
    */
    public static final AssignmentOperator BITWISE_INCLUSIVE_OR =
            new AssignmentOperator(3);

    /**
    * /=
    */
    public static final AssignmentOperator DIVIDE =
            new AssignmentOperator(4);

    /**
    * =
    */
    public static final AssignmentOperator NORMAL =
            new AssignmentOperator(5);

    /**
    * %=
    */
    public static final AssignmentOperator MODULUS =
            new AssignmentOperator(6);

    /**
    * *=
    */
    public static final AssignmentOperator MULTIPLY =
            new AssignmentOperator(7);

    /**
    * &lt;&lt;=
    */
    public static final AssignmentOperator SHIFT_LEFT =
            new AssignmentOperator(8);

    /**
    * &gt;&gt;=
    */
    public static final AssignmentOperator SHIFT_RIGHT =
            new AssignmentOperator(9);

    /**
    * -=
    */
    public static final AssignmentOperator SUBTRACT =
            new AssignmentOperator(10);

    /**
    * Used internally -- you may not create arbitrary assignment operators
    * and may only use the ones provided as static members.
    *
    * @param value The numeric code of the operator.
    */
    private AssignmentOperator(int value) {
        this.value = value;
        op_map.put(names[value], this);
    }

    /**
    * Returns an assignment operator that matches the specified string.
    * @param s the string to be matched.
    * @return the matching assignment operator.
    */
    public static AssignmentOperator fromString(String s) {
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
        if (value < 0 || value > 10) {
            throw new IllegalStateException();
        }
    }

}
