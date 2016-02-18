package cetus.hir;

import java.io.PrintWriter;
import java.util.HashMap;

/**
* Infix operators that accesses righthand side member of lefthand side
* structure, class, or union.
*/
public class AccessOperator extends BinaryOperator {

    private static HashMap<String, AccessOperator> op_map =
            new HashMap<String, AccessOperator>(8);

    private static String[] names = {".", "->", ".*", "->*"};

    /**
    * .
    */
    public static final AccessOperator MEMBER_ACCESS = new AccessOperator(0);

    /**
    * -&gt;
    */
    public static final AccessOperator POINTER_ACCESS = new AccessOperator(1);

    /**
    * .*
    */
    public static final AccessOperator MEMBER_DEREF_ACCESS =
            new AccessOperator(2);

    /**
    * -&gt;*
    */
    public static final AccessOperator POINTER_MEMBER_ACCESS =
            new AccessOperator(3);

    /**
    * Used internally -- you may not create arbitrary assignment operators
    * and may only use the ones provided as static members.
    *
    * @param value The numeric code of the operator.
    */
    private AccessOperator(int value) {
        this.value = value;
        op_map.put(names[value], this);
    }

    /**
    * Returs an <tt>AccessOperator</tt> that matches the specified string.
    * @param s the string to be matched.
    * @return the matching access operator or null if match fails.
    */
    public static AccessOperator fromString(String s) {
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
        if (value < 0 || value > 3) {
            throw new IllegalStateException();
        }
    }

}
