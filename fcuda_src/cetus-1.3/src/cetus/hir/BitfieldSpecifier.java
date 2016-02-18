package cetus.hir;

import java.io.PrintWriter;

/**
* BitfieldSpecifier represents the bit field declared in a <tt>struct</tt>
* declaration.
*/
public class BitfieldSpecifier extends Specifier {

    private Expression bit;

    /**
    * Constructs a bit field specifier from the given bit expression.
    * @param e the expression to be used as a bit field.
    */
    public BitfieldSpecifier(Expression e) {
        bit = e;
    }

    /** Prints the specifier on the specified print writer. */
    public void print(PrintWriter o) {
        o.print(" : ");
        bit.print(o);
    }

}
