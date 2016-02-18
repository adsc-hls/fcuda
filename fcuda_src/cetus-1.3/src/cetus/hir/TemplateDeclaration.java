package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.List;

/** This class is not supported */
public class TemplateDeclaration extends Declaration {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = TemplateDeclaration.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private List params;

    public TemplateDeclaration(List params, Declaration decl) {
        super(1);
        object_print_method = class_print_method;
        this.params = params;
        children.add(decl);
        decl.setParent(this);
    }

    /**
    * Prints a template to a stream.
    *
    * @param d The template to print.
    * @param o The writer on which to print the namespace.
    */
    public static void defaultPrint(TemplateDeclaration d, PrintWriter o) {
        o.print("template < ");
        PrintTools.printListWithComma(d.params, o);
        o.println(" > ");
        d.getDeclaration().print(o);
    }

    public Declaration getDeclaration() {
        return (Declaration)children.get(0);
    }

    public List<IDExpression> getDeclaredIDs() {
        return getDeclaration().getDeclaredIDs();
    }

}
