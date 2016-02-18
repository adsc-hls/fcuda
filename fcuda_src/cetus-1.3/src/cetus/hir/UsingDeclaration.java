package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/** This class is not supported */
public class UsingDeclaration extends Declaration {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = UsingDeclaration.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private IDExpression expr;

    public UsingDeclaration(IDExpression expr) {
        object_print_method = class_print_method;
        this.expr = expr;
    }

    public static void defaultPrint(UsingDeclaration d, PrintWriter o) {
        o.print("using ");
        d.expr.print(o);
        o.print(";");
    }

    @SuppressWarnings("unchecked")
    public List<IDExpression> getDeclaredIDs() {
        return (List<IDExpression>) empty_list;
    }

}
