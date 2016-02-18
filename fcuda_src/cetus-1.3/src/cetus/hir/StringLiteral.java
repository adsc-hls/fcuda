package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** Represents a string literal in the program. */
public class StringLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = StringLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private String value;

    /** Constructs a string literal with the specified string value. */
    public StringLiteral(String s) {
        object_print_method = class_print_method;
        value = s;
    }

    /** Returns a clone of the string literal. */
    @Override
    public StringLiteral clone() {
        StringLiteral o = (StringLiteral)super.clone();
        o.value = value;
        return o;
    }

    /**
    * Prints a literal to a stream.
    *
    * @param l The literal to print.
    * @param o The writer on which to print the literal.
    */
    public static void defaultPrint(StringLiteral l, PrintWriter o) {
        o.print("\"" + l.value + "\"");
    }

    /** Returns a string representation of the string literal. */
    @Override
    public String toString() {
        return ("\"" + value + "\"");
    }

    /** Compares the string literal with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && value.equals(((StringLiteral)o).value));
    }

    /** Returns the value of the string literal. */
    public String getValue() {
        return value;
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

    /** Sets the value of the literal with the specified string value. */
    public void setValue(String value) {
        this.value = value;
    }

    /**
    * Removes outer quotes from the string; this class automatically
    * prints quotes around the string so it does not need to store them.
    * Thus, if you create a StringLiteral with a String that already has
    * quotes around it, you will have double quotes, and may need to call
    * this method.  The method has no effect if there are no quotes.
    */
    public void stripQuotes() {
        String quote = "\"";
        if (value.startsWith(quote) && value.endsWith(quote)) {
            value = value.substring(1, value.length() - 1);
        }
    }

}
