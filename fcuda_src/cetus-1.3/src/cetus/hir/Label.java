package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a label for use with goto statements.
*/
public class Label extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Label.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Creates a new label with the specified name ID.
    *
    * @param name the name of the label.
    * @throws IllegalArgumentException if <b>name</b> is null.
    * @throws NotAnOrphanException if <b>name</b> has a parent.
    */
    public Label(IDExpression name) {
        object_print_method = class_print_method;
        if (name != null && !(name instanceof NameID)) {
            name = new NameID(name.toString());
        }
        addChild(name);
    }

    /**
    * Prints a label to a stream.
    *
    * @param l The label to print.
    * @param o The writer on which to print the label.
    */
    public static void defaultPrint(Label l, PrintWriter o) {
        l.getName().print(o);
        o.print(":");
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
    * Returns the string for this Label
    */
    public IDExpression getName() {
        return (IDExpression)children.get(0);
    }

    /**
    * Sets the name of the label with the specified name.
    *
    * @param name - name of the label.
    * @throws IllegalArgumentException if <b>name</b> is null.
    * @throws NotAnOrphanException if <b>name</b> has a parent.
    */
    public void setName(IDExpression name) {
        setChild(0, name);
    }

}
