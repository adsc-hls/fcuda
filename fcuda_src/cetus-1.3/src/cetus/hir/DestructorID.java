package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;

/**
* Represents the name of a C++ destructor. This class is no longer supported.
*/
public class DestructorID extends IDExpression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = DestructorID.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    public DestructorID(IDExpression class_name) {
        super(false);
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>(1);
        children.add(class_name);
        class_name.setParent(this);
    }

    @Override
    public DestructorID clone() {
        DestructorID o = (DestructorID)super.clone();
        return o;
    }

    /**
    * Prints a destructor to a stream.
    *
    * @param d The destructor to print.
    * @param o The writer on which to print the destructor.
    */
    public static void defaultPrint(DestructorID d, PrintWriter o) {
        o.print("~");
        d.getClassName().print(o);
    }

    public IDExpression getClassName() {
        return (IDExpression)children.get(0);
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
