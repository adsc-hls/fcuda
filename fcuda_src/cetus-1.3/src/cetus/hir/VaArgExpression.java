package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/** Represents __builtin_va_arg() operation in C programs. */
public class VaArgExpression extends Expression implements Intrinsic {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = VaArgExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private List specs;

    /**
    * Constructs a va_arg expression with the specified expression and specs.
    *
    * @param expr the operand expression.
    * @param pspecs the list of specifiers.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    @SuppressWarnings("unchecked")
    public VaArgExpression(Expression expr, List pspecs) {
        object_print_method = class_print_method;
        addChild(expr);
        specs = new ArrayList(pspecs);
    }

    /**
    * Prints a __builtin_va_arg expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(VaArgExpression e, PrintWriter o) {
        o.print("__builtin_va_arg(");
        e.getExpression().print(o);
        o.print(",");
        PrintTools.printListWithSpace(e.specs, o);
        o.print(")");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("__builtin_va_arg(");
        sb.append(getExpression());
        sb.append(",");
        sb.append(PrintTools.listToString(specs, " "));
        sb.append(")");
        return sb.toString();
    }

    @Override
    protected int hashCode(int h) {
        h = hashCode("__builtin_va_arg(", h);
        h = getExpression().hashCode(h);
        h = 31 * h + ',';
        h = hashCode(specs, " ", h);
        h = 31 * h + ')';
        return h;
    }

    /**
    * Returns the expression.
    *
    * @return the expression or null if this sizeof operator is
    *   being applied to a type.
    */
    public Expression getExpression() {
        return (Expression)children.get(0);
    }

    /**
    * Returns the type argument which is also the type of return value.
    *
    * @return the list of specifiers.
    */
    public List getSpecifiers() {
        return specs;
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
    * Compares the va_arg expression with the specified object for equality.
    */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) &&
                specs.equals(((VaArgExpression)o).specs));
    }

}
