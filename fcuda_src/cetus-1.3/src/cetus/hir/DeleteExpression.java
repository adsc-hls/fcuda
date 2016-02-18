package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/** This class is not supported. */
public class DeleteExpression extends Expression {

    private static Method class_print_method;

    /**
    * if global == true, it prints "::" ahead of delelte expression
    */
    protected static boolean global;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = DeleteExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private boolean array = false;

    public DeleteExpression(Expression expr) {
        this.array = false;
        children.add(expr);
        expr.setParent(this);
        global = false;
    }

    public DeleteExpression(boolean array, Expression expr) {
        this.array = array;
        children.add(expr);
        expr.setParent(this);
        global = false;
    }

    /**
    * force printing "::"
    */
    public void setGlobal() {
        global = true;
    }

    /**
    * disable printing "::"
    */
    public void clearGlobal() {
        global = false;
    }

    /**
    * returns true if it is set to be global.
    */
    public boolean isGlobal() {
        return global;
    }

    @Override
    public DeleteExpression clone() {
        DeleteExpression o = (DeleteExpression)super.clone();
        o.array = array;
        o.global = this.global;
        return o;
    }

    /**
    * Prints a delete expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(DeleteExpression e, PrintWriter o) {
        if (global) {
            o.print("::");
        }
        o.print("delete ");
        if (e.array) {
            o.print("[] ");
        }
        e.getExpression().print(o);
    }

    public Expression getExpression() {
        return (Expression)children.get(0);
    }

    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && array == ((DeleteExpression)o).array);
    }

}
