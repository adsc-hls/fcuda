package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** Represents a boolean literal (true or false). */
public class BooleanLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = BooleanLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private boolean value;

    /**
    * Constructs a boolean literal with the specified boolean value.
    *
    * @param value the boolean value.
    */
    public BooleanLiteral(boolean value) {
        object_print_method = class_print_method;
        this.value = value;
    }

    /** Returns a clone of the boolean literal. */
    @Override
    public BooleanLiteral clone() {
        BooleanLiteral o = (BooleanLiteral)super.clone();
        o.value = value;
        return o;
    }

    /**
    * Prints a literal to a stream.
    *
    * @param l The literal to print.
    * @param o The writer on which to print the literal.
    */
    public static void defaultPrint(BooleanLiteral l, PrintWriter o) {
        o.print(Boolean.toString(l.value));
    }

    /** Returns a string representation of the boolean literal. */
    @Override
    public String toString() {
        return Boolean.toString(value);
    }

    /** Compares the boolean literal with the specified object. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && value == ((BooleanLiteral)o).value);
    }

    /**
    * Returns the boolean value.
    *
    * @return the value of this BooleanLiteral.
    */
    public boolean getValue() {
        return value;
    }

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

    /**
    * Set the boolean value of this object.
    */
    public void setValue(boolean value) {
        this.value = value;
    }

}
