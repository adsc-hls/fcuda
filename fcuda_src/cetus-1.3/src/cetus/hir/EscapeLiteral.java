package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** Represents a escape character. */
public class EscapeLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = EscapeLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private char value;

    private String name;

    /** Constructs an escape literal with the specified string name. */
    public EscapeLiteral(String name) {
        char c = name.charAt(2);
        switch (c) {
        case 'a':
            c = '\7';
            break;
        case 'b':
            c = '\b';
            break;
        case 'f':
            c = '\f';
            break;
        case 'n':
            c = '\n';
            break;
        case 'r':
            c = '\r';
            break;
        case 't':
            c = '\t';
            break;
        case 'v':
            c = '\14';
            break;
        case '\\':
            c = '\\';
            break;
        case '?':
            c = '\77';
            break;
        case '\'':
            c = '\'';
            break;
        case '\"':
            c = '\"';
            break;
        case 'x':
            c = (char)Integer.parseInt(name.substring(3, name.length()-1), 16);
            break;
        default:
            if (c <= '7' && c >= '0') {
                c = (char)Integer.parseInt(name.substring(
                        2, name.length()-1), 8);
            } else {
                c = '?';
                PrintTools.printlnStatus(0,
                        "Unrecognized Escape Sequence", name);
            }
            break;
        }
        object_print_method = class_print_method;
        this.name = name;
        this.value = c;
    }

    /** Returns a clone of the escape literal. */
    @Override
    public EscapeLiteral clone() {
        EscapeLiteral o = (EscapeLiteral)super.clone();
        o.value = value;
        o.name = name;
        return o;
    }

    /**
    * Prints a literal to a stream.
    *
    * @param l The literal to print.
    * @param o The writer on which to print the literal.
    */
    public static void defaultPrint(EscapeLiteral l, PrintWriter o) {
        o.print(l.name);
    }

    /** Returns a string representation of the escape literal. */
    @Override
    public String toString() {
        return name;
    }

    /** Compares the escape literal with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && value == ((EscapeLiteral)o).value);
    }

    /** Returns the character value of the escape literal. */
    public char getValue() {
        return value;
    }

    /** Returns the hash code of the escape literal. */
    @Override
    public int hashCode() {
        return toString().hashCode();
    }

    /**
    * Overrides the class print method, so that all subsequently created
    * objects will use the supplied method.
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
