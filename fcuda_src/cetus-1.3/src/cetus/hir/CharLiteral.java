package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** Represents a character literal in the program. */
public class CharLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = CharLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private char value;

    /**
    * Constructs a char literal with the specified character value.
    *
    * @param value the character value.
    */
    public CharLiteral(char value) {
        object_print_method = class_print_method;
        this.value = value;
    }

    /** Returns a clone of the char literal. */
    @Override
    public CharLiteral clone() {
        CharLiteral o = (CharLiteral)super.clone();
        o.value = value;
        return o;
    }

    /**
    * Prints a literal to a stream.
    *
    * @param l The literal to print.
    * @param o The writer on which to print the literal.
    */
    public static void defaultPrint(CharLiteral l, PrintWriter o) {
        o.print("'" + l.value + "'");
    }

    /** Returns a string representation of the char literal. */
    @Override
    public String toString() {
        return ("'" + value + "'");
    }

    /** Compares the char literal with the given object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && value == ((CharLiteral)o).value);
    }

    /** Returns the character value of the char literal. */
    public char getValue() {
        return value;
    }

    /** Returns the hash code of the char literal. */
    @Override
    public int hashCode() {
        return toString().hashCode();
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

    /** Sets the value of the literal with the specified character. */
    public void setValue(char value) {
        this.value = value;
    }

}
