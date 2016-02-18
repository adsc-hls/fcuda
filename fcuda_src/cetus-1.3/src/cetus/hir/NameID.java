package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* <b>NameID</b> is introduced to separate the uses of {@link Identifier} for
* better IR consistency. <b>Identifier</b> is used only for variables which
* are part of expression tree,
* whereas <b>NameID</b> is for other types of traversable objects that do not
* need to refer to the declared variables. One suggestion for pass writers is
* that <b>avoid</b> using <b>Identifier</b> if the object is not going
* to have any access to the declared symbol. This means an <b>Identifier</b>
* object is created only through the constructor
* {@link Identifier#Identifier(Symbol)}.
* Following example explicates the * uses of these two types of
* <b>IDExpression</b>.
* <pre>
*   ...
*   int a = b+1;  // a is a NameID object, b is an Identifier object.
*   a += ...      // a is an Identifier object.
*   ...
* </pre>
*/
public class NameID extends IDExpression {

    /** Default print method for the class */
    private static Method class_print_method;

    /** Static setup */
    static {
        try {
            class_print_method = NameID.class.getMethod("defaultPrint",
                    new Class<?>[] {NameID.class, PrintWriter.class});
        } catch(NoSuchMethodException e) {
            throw new InternalError(e.getMessage());
        }
    }

    /** The name of the ID */
    private String name;

    /**
    * Constructs a new named ID with the given string name.
    * 
    * @param name the name of this named ID.
    */
    public NameID(String name) {
        super(false);
        this.name = name;
        object_print_method = class_print_method;
    }

    /**
    * Returns the string name of this named ID.
    *
    * @return the string name.
    */
    @Override
    public String getName() {
        return name;
    }

    /**
    * Sets the name of this ID with the given new name.
    * 
    * @param new_name the new string name for this named ID.
    */
    public void setName(String new_name) {
        name = new_name;
    }

    /**
    * Prints the name of this ID to the given writer.
    *
    * @param i the named ID to be printed.
    * @param o the print writer. 
    */
    public static void defaultPrint(NameID i, PrintWriter o) {
        o.print(i.name);
    }

    /**
    * Returns a clone of this named ID.
    *
    * @return the cloned ID.
    */
    @Override
    public NameID clone() {
        NameID o = (NameID)super.clone();
        o.name = this.name;
        return o;
    }

    /**
    * Returns a string representation of this named ID.
    *
    * @return the string representation.
    */
    @Override
    public String toString() {
        return name;
    }

    /**
    * Checks if the given object <b>o</b> is equal to this named ID. For an
    * internal reason, it is possible for a <b>NameID</b> object to be equal to
    * an <b>Identifier</b> object; they have been implemented throuhg
    * <b>Identifier</b> before.
    *
    * @param o the object to be checked for equality.
    * @return true if {@code o.hashCode() == this.hashCode()}.
    */
    @Override
    public boolean equals(Object o) {
        return (o != null &&
                o instanceof IDExpression && o.toString().equals(name));
    }

    /**
    * Returns the hash code of this named ID.
    *
    * @return the string name's hash code.
    */
    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    protected int hashCode(int h) {
        return hashCode(name, h);
    }

}
