package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** Represents a float-type literal in the program. */
public class FloatLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = FloatLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private double value;

    /** Suffix for the size of the float literal */
    private String suffix;

    /** Constructs a float literal with the specified value. */
    public FloatLiteral(double value) {
        object_print_method = class_print_method;
        this.value = value;
        this.suffix = "";
    }

    /** Constructs a float literal with the specified value and suffix. */
    public FloatLiteral(double value, String suffix) {
        object_print_method = class_print_method;
        this.value = value;
        this.suffix = suffix;
    }

    /** Returns a clone of the float literal. */
    @Override
    public FloatLiteral clone() {
        FloatLiteral o = (FloatLiteral)super.clone();
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
    public static void defaultPrint(FloatLiteral l, PrintWriter o) {
        o.print(Double.toString(l.value));
        o.print(l.suffix);
    }

    /** Returns a string representation of the float literal. */
    @Override
    public String toString() {
        return (Double.toString(value) + suffix);
    }

    /** Compare the float literal with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) &&
                value == ((FloatLiteral)o).value &&
                suffix.equals(((FloatLiteral)o).suffix));
    }

    /** Returns the numeric value of the float literal. */
    public double getValue() {
        return value;
    }

    /** Returns the hash code of the float literal. */
    @Override public int hashCode() {
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

    /** Sets the value of the float literal with the specified value. */
    public void setValue(double value) {
        this.value = value;
    }

    /** Sets the suffix of the float literal with the specified string. */
    public void setSuffix(String suffix) {
        this.suffix = suffix;
    }

}
