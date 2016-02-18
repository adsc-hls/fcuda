package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** Represents an integer literal in the program. */
public class IntegerLiteral extends Literal {

    private static Method class_print_method;

    //specifically used to handle 
    private static Method hex_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = IntegerLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
            hex_print_method = params[0].getMethod("printHex", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private long value;

    private String suffix;

    /** Constructs an integer literal with the specified numeric value. */
    public IntegerLiteral(long value) {
        object_print_method = class_print_method;
        this.value = value;
        this.suffix = "";
    }

    /** Constructs an integer literal with the specified value and suffix. */
    public IntegerLiteral(long value, String suffix) {
        object_print_method = class_print_method;
        this.value = value;
        this.suffix = suffix;
    }

    /**
    * Construct an IntegerLiteral from a string and set the proper print method
    */
    public IntegerLiteral(String _int_str) {
        //String  D = "[0-9]";
        //String  L = "[a-zA-Z_]";
        String H = "[a-fA-F0-9]";
        //String  E = "[Ee][+-]?(" + D + ")+";
        //String  FS  = "(f|F|l|L)";
        String IS = "(u|U|l|L)*";
        String const1 = "0[xX](" + H + ")+(" + IS + ")?";
        //String const2 = "0(" + D + ")+(" + IS + ")?";
        //String const3 = "(" + D + ")+(" + IS+ ")?";
        //String const4 = L + "?'(\\.|[^\\'])+'";  //weird
        //String const5 = "(" + D + ")+(" + E + ")(" + FS + ")?";
        //String const6 = "(" + D + ")*.(" + D + ")+(" + E + ")?" + FS + "?";
        //String const7 = "(" + D + ")+.(" + D + ")*(" + E + ")?" + FS + "?";
        //String  int_exp = const1 + "|" + const2 + "|" + const3;
        //String  float_exp = const5 + "|" + const6 + "|" + const7;
        String int_hex_exp = const1;
        //String  int_dec_exp = const2 + "|" + const3;
        if (_int_str.matches(int_hex_exp) == true) {
            //set the print method
            //this.setPrintMethod(this.printHex);
            object_print_method = hex_print_method;
        } else {
            object_print_method = class_print_method;
        }
        this.value = (Integer.decode(_int_str)).intValue();
        this.suffix = "";
    }

    /** Returns a clone of the integer literal. */
    @Override
    public IntegerLiteral clone() {
        IntegerLiteral o = (IntegerLiteral)super.clone();
        o.value = value;
        o.suffix = suffix;
        return o;
    }

    /**
    * Prints a literal to a stream.
    *
    * @param l The literal to print.
    * @param o The writer on which to print the literal.
    */
    public static void defaultPrint(IntegerLiteral l, PrintWriter o) {
        o.print(Long.toString(l.value));
        o.print(l.suffix);
    }

    /** Returns a string representation of the integer literal. */
    @Override
    public String toString() {
        return (Long.toString(value) + suffix);
    }

    /** Compares the integer literal with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) &&
                value == ((IntegerLiteral) o).value &&
                suffix.equals(((IntegerLiteral)o).suffix));
    }

    /** Returns the numeric value of the integer literal. */
    public long getValue() {
        return value;
    }

    /** Returns the hash code of the integer literal. */
    @Override
    public int hashCode() {
        return toString().hashCode();
    }

    /** Prints the integer literal in the hexadecimal format. */
    public static void printHex(IntegerLiteral l, PrintWriter o) {
        o.print(Long.toHexString(l.value));
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
    * Sets the value of the integer literal with the specified numeric value.
    */
    public void setValue(long value) {
        this.value = value;
    }

}
