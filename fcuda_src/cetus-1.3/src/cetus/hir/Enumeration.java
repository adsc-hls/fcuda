package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
* <b>Enumeration</b> represents a C or C++ enumeration.
*/
public class Enumeration extends Declaration {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Enumeration.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** The name of the enumeration expressed in terms of an ID */
    private IDExpression name;

    /** The specifier created from this enumeration */
    private UserSpecifier specifier;

    /**
    * Creates an enumeration.
    *
    * @param name The name of the enumeration.
    * @param declarators A list of declarators to use as the enumerators.
    *   For enumerations that are not consecutive, initializers should be
    *   placed on the declarators.
    */
    public Enumeration(IDExpression name, List declarators) {
        super(declarators.size());
        object_print_method = class_print_method;
        if (name == null) {
            throw new IllegalArgumentException();
        }
        this.name = name;
        this.specifier =
            new UserSpecifier(new NameID("enum " + name.toString()));
        try {
            int declarators_size = declarators.size();
            for (int i = 0; i < declarators_size; i++) {
                Declarator d = (Declarator)declarators.get(i);
                children.add(d);
                d.setParent(this);
            }
        } catch(ClassCastException ex) {
            throw new IllegalArgumentException();
        }
    }

    /**
    * Prints an enumeration to a stream.
    *
    * @param e The enumeration to print.
    * @param o The writer on which to print the enumeration.
    */
    public static void defaultPrint(Enumeration e, PrintWriter o) {
        e.specifier.print(o);
        o.print(" { ");
        PrintTools.printListWithComma(e.children, o);
        o.print(" }");
    }

    /* Declaration.getDeclaredIDs() */
    public List<IDExpression> getDeclaredIDs() {
        List<IDExpression> ret = new ArrayList<IDExpression>(children.size());
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = children.get(i);
            if (child instanceof Declarator) {
                ret.add(((Declarator)child).getID());
            }
        }
        return ret;
    }

    /**
    * Returns the name ID of this enumeration.
    *
    * @return the identifier holding the enum name.
    */
    public IDExpression getName() {
        return name;
    }

    /**
    * Returns the specifier created from this enumeration.
    *
    * @return the user specifier {@code enum name}.
    */
    public Specifier getSpecifier() {
        return specifier;
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

}
