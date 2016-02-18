package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.ArrayList;

/**
* Initializer holds initial values for the associated variable declarator.
* NOTE: Subclasses of this class are not used in C.
*/
public class Initializer implements Cloneable, Traversable {

    // two types of initializers - value initializers (single
    // values and array of values) and constructors
    // = Expression
    // = { List of Expressions }
    // What about...?
    // my_class y(1, 2);

    /** Class print method */
    private static Method class_print_method;

    /** Object-specific print method */
    private Method object_print_method;

    /** Parent traversable object */
    protected Traversable parent;

    /** List of children */
    protected List<Traversable> children;

    /** Flags for indicating it is a list */
    private boolean is_list;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Initializer.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Constructs a new empty initializer
    */
    protected Initializer() {
        children = new ArrayList<Traversable>(1);
    }

    /**
    * Constructs a new initializer with the specified initializing value.
    * @param value the initializing value expression.
    */
    public Initializer(Expression value) {
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>(1);
        addChild(value);
        is_list = false;
    }

    /**
    * Constructs a new initializer with the specified list of values.
    * @param values the list of initializing values.
    */
    public Initializer(List values) {
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>(values.size());
        for (Object value : values) {
            if (value instanceof Expression || value instanceof Initializer) {
                addChild((Traversable) value);
            } else {
                throw new IllegalArgumentException(this.getClass().getName());
            }
        }
        is_list = true;
    }

    /**
    * Adds the specified traversable object to the list of children.
    * @param t the traversable object to be added.
    */
    protected void addChild(Traversable t) {
        if (t.getParent() != null) {
            throw new NotAnOrphanException(this.getClass().getName());
        }
        children.add(t);
        t.setParent(this);
    }

    @Override
    public Initializer clone() {
        Initializer o = null;
        try {
            o = (Initializer) super.clone();
        } catch(CloneNotSupportedException e) {
            throw new InternalError();
        }
        o.object_print_method = object_print_method;
        o.parent = null;
        o.children = new ArrayList<Traversable>(children.size());
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = children.get(i);
            if (child instanceof Expression) {
                o.addChild(((Expression)child).clone());
            } else {
                o.addChild(((Initializer)child).clone());
            }
        }
        return o;
    }

    /**
    * Prints an initializer to the specified writer.
    * @param i The initializer to print.
    * @param o The writer on which to print the initializer.
    */
    public static void defaultPrint(Initializer i, PrintWriter o) {
        if (!(i.parent instanceof Initializer)) {
            o.print(" = ");
        }
        if (i.is_list) {
            o.print(" { ");
            PrintTools.printListWithComma(i.children, o);
            o.print(" } ");
        } else {
            PrintTools.printList(i.children, o);
        }
    }

    /**
    * Converts this initializer to a string by calling the default print method.
    * All sub classes will be using this method unless specialized.
    */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        if (!(parent instanceof Initializer)) {
            sb.append(" = ");
        }
        if (is_list) {
            sb.append(" { ");
            sb.append(PrintTools.listToString(children, ", "));
            sb.append(" } ");
        } else {
            sb.append(PrintTools.listToString(children, ""));
        }
        return sb.toString();
    }

    public List<Traversable> getChildren() {
        return children;
    }

    public Traversable getParent() {
        return parent;
    }

    /** Prints an initializer object by calling its default print method. */
    public void print(PrintWriter o) {
        if (object_print_method == null) {
            return;
        }
        try {
            object_print_method.invoke(null, new Object[] {this, o});
        } catch(IllegalAccessException e) {
            throw new InternalError();
        } catch(InvocationTargetException e) {
            throw new InternalError();
        }
    }

    public void removeChild(Traversable child) {
        throw new UnsupportedOperationException(
                "Initializers do not support removal of arbitrary children.");
    }

    public void setChild(int index, Traversable t) {
        children.get(index).setParent(null);
        t.setParent(this);
        children.set(index, t);
    }

    /**
    * Overrides the class print method, so that all subsequently
    * created objects will use the supplied method.
    * @param m The new print method.
    */
    static public void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    public void setParent(Traversable t) {
        parent = t;
    }

    /**
    * Overrides the print method for this object only.
    * @param m The new print method.
    */
    public void setPrintMethod(Method m) {
        object_print_method = m;
    }

}
