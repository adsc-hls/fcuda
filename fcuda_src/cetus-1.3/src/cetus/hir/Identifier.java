package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* <b>Identifier</b> represents a valid C identifier which has a matching
* variable declarator. It is important to notice that construction of a new
* Identifier object with an arbitrary string is not recommended at all because
* of possible inconsistency. For this reason such constructors are protected.
* For pass writers accustomed to such usage may still be able to create a
* <b>string</b> identifier through {@link SymbolTools#getOrphanID(String)}.
* However, the correctness of any analysis/transformation after introducing such
* "orphan" identifier is not guaranteed.
*/
public class Identifier extends IDExpression {

    /** Default print method for Identifier object */
    private static Method class_print_method;

    /** Assigns default print method */
    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Identifier.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError(e.getMessage());
        }
    }

    /**
    * Fall-back string name for this identifier; it is used only when intended.
    */
    private String name;

    /** Reference to the relevant symbol object. */
    private Symbol symbol;

    /**
    * Returns a new <b>incomplete</b> identifier with the given string name.
    * It is incomplete in that it does not have access to its relevant
    * variable declaration/declarator. This constructor is used only internally.
    * Pass writers are strongly recommended to construct an identifier out of
    * the relevant symbol object.
    *
    * @param name the raw string name.
    */
    protected Identifier(String name) {
        super(false);
        object_print_method = class_print_method;
        this.name = name;
    }

    /**
    * Constructs and returns a new <b>Identifier</b> with the given
    * <b>Symbol</b> object.
    *
    * @param symbol the relevant symbol object, which is typically a varaible
    * declarator.
    */
    public Identifier(Symbol symbol) {
        super(false);
        object_print_method = class_print_method;
        this.symbol = symbol;
    }

    /**
    * Returns a new <b>incomplete</b> identifier with the given qualifier and
    * name.
    */
    protected Identifier(boolean global, String name) {
        super(global);
        object_print_method = class_print_method;
        this.name = name;
    }

    /**
    * Returns a clone of this identifier. For best consistency, the cloned
    * identifier should be used in a scope in which the original symbol is
    * accessible since the reference to the relevant symbol is also copied.
    */
    @Override
    public Identifier clone() {
        Identifier o = (Identifier)super.clone();
        o.symbol = this.symbol;
        o.name = this.name;     // if it exists.
        return o;
    }

    /**
    * Prints an identifier to a stream.
    *
    * @param i The identifier to print.
    * @param o The writer on which to print the identifier.
    */
    public static void defaultPrint(Identifier i, PrintWriter o) {
        if (i.typename) {
            o.print("typename ");       // not used in C
        }
        if (i.global) {
            o.print("::");      // not used in C
        }
        o.print(i.getName());
    }

    /**
    * Returns a string representation of this identifier (= {@link #getName()}).
    */
    @Override
    public String toString() {
        return getName();
    }

    /**
    * Checks if the given object <b>o</b> is equal to this identifier.
    *
    * @param o the object to be compared with.
    * @return true if the given object is an IDExpression and string comparison
    * matches.
    */
    @Override
    public boolean equals(Object o) {
        return (o != null &&
                o instanceof IDExpression &&
                o.toString().equals(getName()));
    }

    /** Returns the hashcode of this identifier */
    @Override
    public int hashCode() {
        // see the IDExpression grandparent class for the toString
        // implementation
        return getName().hashCode();
    }

    @Override
    protected int hashCode(int h) {
        return hashCode(getName(), h);
    }

    /**
    * Returns the string name of this identifier.
    *
    * @return the string name of the relevant symbol object, or locally stored
    * <b>fall-back</b> name if intended.
    */
    @Override
    public String getName() {
        if (symbol != null) {
            return symbol.getSymbolName();
        } else {
            return name;        // This is only for inconsistent program.
        }
    }

    /**
    * Sets the symbol field with the given symbol object. This is used only
    * internally.
    *
    * @param symbol the Symbol object to be linked against.
    */
    protected void setSymbol(Symbol symbol) {
        this.symbol = symbol;
        name = null;
    }

    /**
    * Returns the {@link Symbol} object linked with this identifier.
    */
    public Symbol getSymbol() {
        return symbol;
    }

}
