package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/** Represents a __builtin_offsetof() operator. */
public class OffsetofExpression extends Expression implements Intrinsic {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = OffsetofExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private List specs;

    /**
    * Constructs an offsetof expression with the specified specifiers and
    * the expression.
    *
    * @param pspecs the list of specifiers.
    * @param expr the operand expression.
    * @throws IllegalArgumentException if <b>expr</b> is invalid.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    @SuppressWarnings("unchecked")
    public OffsetofExpression(List pspecs, Expression expr) {
        object_print_method = class_print_method;
        addChild(expr);
        specs = new ArrayList(pspecs);
    }

    /**
    * Prints a __builtin_offsetof expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(OffsetofExpression e, PrintWriter o) {
        o.print("__builtin_offsetof(");
        PrintTools.printListWithSeparator(e.specs, o, " ");
        o.print(",");
        e.getExpression().print(o);
        o.print(")");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("__builtin_offsetof(");
        sb.append(PrintTools.listToString(specs, " "));
        sb.append(",");
        sb.append(getExpression());
        sb.append(")");
        return sb.toString();
    }

    @Override
    protected int hashCode(int h) {
        h = hashCode("__builtin_offsetof(", h);
        h = hashCode(specs, " ", h);
        h = 31 * h + ',';
        h = getExpression().hashCode(h);
        h = 31 * h + ')';
        return h;
    }

    /**
    * Returns the expression operand.
    *
    * @return the expression that is the second parameter
    * to this function (member-designator)
    */
    public Expression getExpression() {
        return (Expression)children.get(0);
    }

    /** Returns the list of specifiers. */
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

    /** Compares the expression with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) &&
                specs.equals(((OffsetofExpression)o).specs));
    }

}
