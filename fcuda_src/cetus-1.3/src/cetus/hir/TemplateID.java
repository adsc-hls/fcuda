package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.List;
import java.util.ArrayList;

/** This class is no longer supported */
public class TemplateID extends IDExpression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = TemplateID.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private String name;

    public TemplateID(String name) {
        super(false);
        object_print_method = class_print_method;
        this.name = name;
    }

    public TemplateID(String name, List args) {
        super(false);
        object_print_method = class_print_method;
        this.name = name;
        children = new ArrayList<Traversable>(args.size());
        for (Object o : args) {
            children.add((Traversable)o);
            ((Traversable)o).setParent(this);
        }
    }

    @Override
    public TemplateID clone() {
        TemplateID o = (TemplateID)super.clone();
        o.name = name;
        return o;
    }

    /**
    * Prints an identifier to a stream.
    *
    * @param i The identifier to print.
    * @param o The writer on which to print the identifier.
    */
    public static void defaultPrint(TemplateID i, PrintWriter o) {
        if (i.global) {
            o.print("::");
        }
        o.print(i.name);
        o.print("<");
        // template arguments
        o.print(">");
    }

    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && name.equals(((TemplateID)o).name));
    }

    public int hashCode() {
        return toString().hashCode();
    }

}
