package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/** This class is no longer supported */
public class QualifiedID extends IDExpression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = QualifiedID.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    @SuppressWarnings("unchecked")
    public QualifiedID(List id_expr_list) {
        super(false);
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>(id_expr_list);
    }

    @SuppressWarnings("unchecked")
    public QualifiedID(boolean global, List id_expr_list) {
        super(global);
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>(id_expr_list);
    }

    /**
    * This method returns the list of IDs in the qualified ID
    */
    public List getIDExpressionList() {
        return children;
    }

    @Override
    public QualifiedID clone() {
        QualifiedID o = (QualifiedID)super.clone();
        return o;
    }

    /**
    * Prints an identifier to a stream.
    *
    * @param i The identifier to print.
    * @param o The writer on which to print the identifier.
    */
    public static void defaultPrint(QualifiedID i, PrintWriter o) {
        if (i.typename) {
            o.print("typename ");
        }
        if (i.global) {
            o.print("::");
        }
        PrintTools.printListWithSeparator(i.children, o, "::");
    }

}
