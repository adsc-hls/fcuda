package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a statement that contains a declaration as the only child, and
* appears within a compound statement.
*/
public class DeclarationStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = DeclarationStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create a declaration statement given a declaration.
    *
    * @param decl The declaration part of the statement.
    * @throws IllegalArgumentException if <b>decl</b> is null.
    * @throws NotAnOrphanException if <b>decl</b> has a parent.
    */
    public DeclarationStatement(Declaration decl) {
        object_print_method = class_print_method;
        addChild(decl);
    }

    /**
    * Prints a statement to a stream.
    *
    * @param s The statement to print.
    * @param o The writer on which to print the statement.
    */
    public static void defaultPrint(DeclarationStatement s, PrintWriter o) {
        s.getDeclaration().print(o);
        o.print(";");
    }

    /**
    * Returns the declaration part of the statement.
    *
    * @return the declaration part of the statement.
    */
    public Declaration getDeclaration() {
        return (Declaration)children.get(0);
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

    /** Returns a clone of the declaration statement. */
    @Override
    public DeclarationStatement clone() {
        DeclarationStatement ds = (DeclarationStatement)super.clone();
        return ds;
    }

}
