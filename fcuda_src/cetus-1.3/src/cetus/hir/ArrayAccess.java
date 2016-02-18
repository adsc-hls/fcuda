package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
* Represents the access of an array or pointer variable: array[x][y]...
*/
public class ArrayAccess extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class <?>[2];
        try {
            params[0] = ArrayAccess.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Creates an array access with a single index expression.
    *
    * @param array An expression evaluating to an address.
    * @param index The expression with which to index the array.
    * @throws NotAnOrphanException if <b>array</b> or <b>index</b> has a parent
    * object.
    */
    public ArrayAccess(Expression array, Expression index) {
        super(2);
        object_print_method = class_print_method;
        addChild(array);
        addChild(index);
    }

    /**
    * Creates an array access with multiple index expressions.
    *
    * @param array An expression evaluating to an address.
    * @param indices A list of expressions with which to index the array.
    * @throws NotAnOrphanException if <b>array</b> or an element of
    * <b>indices</b> has a parent object.
    */
    public ArrayAccess(Expression array, List indices) {
        super(indices.size() + 1);
        object_print_method = class_print_method;
        addChild(array);
        setIndices(indices);
    }

    /**
    * Inserts a new index at the end of the index list, increasing the dimension
    * of the array access.
    *
    * @param expr the new index expression to be inserted.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void addIndex(Expression expr) {
        addChild(expr);
    }

    /** Returns a clone of the array access */
    @Override
    public ArrayAccess clone() {
        ArrayAccess o = (ArrayAccess)super.clone();
        return o;
    }

    /**
    * Prints an array access expression to a stream.
    *
    * @param e The array access to print.
    * @param o The writer on which to print the array access.
    */
    public static void defaultPrint(ArrayAccess e, PrintWriter o) {
        e.getArrayName().print(o);
        o.print("[");
        PrintTools.printListWithSeparator(
                e.children.subList(1, e.children.size()), o, "][");
        o.print("]");
    }

    /**
    * Returns a string representation of the array access. The returned string
    * is equivalent to the printed stream through {@link #defaultPrint}.
    * @return the string representation.
    */
    @Override public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append(getArrayName());
        sb.append("[");
        sb.append(PrintTools.listToString(
                children.subList(1, children.size()), "]["));
        sb.append("]");
        return sb.toString();
    }

    protected int hashCode(int h) {
        h = ((Expression)children.get(0)).hashCode(h);
        h = 31 * h + '[';
        h = hashCode(children.subList(1, children.size()), "][", h);
        h = 31 * h + ']';
        return h;
    }

    /**
    * Returns the expression being indexed. Most often, the expression will be
    * an Identifier, but any expression that evaluates to an address is allowed.
    *
    * @return the expression being indexed.
    */
    public Expression getArrayName() {
        return (Expression)children.get(0);
    }

    /**
    * Gets the n-th index expression of this array access.
    *
    * @param n The position of the index expression.
    * @throws IndexOutOfBoundsException if there is no expression at that
    *       position.
    * @return the nth index expression.
    */
    public Expression getIndex(int n) {
        return (Expression)children.get(n + 1);
    }

    /**
    * Returns the list of indices in the array access.
    * @return the list of indices.
    */
    public List<Expression> getIndices() {
        List<Expression> ret = new ArrayList<Expression>(children.size() - 1);
        for (int i = 0; i < getNumIndices(); ++i) {
            ret.add(getIndex(i));
        }
        return ret;
    }

    /**
    * Returns the number of index expressions used in this array access.
    *
    * @return the number of index expressions.
    */
    public int getNumIndices() {
        return children.size() - 1;
    }

    /**
    * Sets the nth index expression of this array access.
    *
    * @param n The position of the index expression.
    * @param expr The expression to use for the index.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    * @throws IndexOutOfBoundsException if there is no expression at that
    *       position.
    */
    public void setIndex(int n, Expression expr) {
        setChild(n + 1, expr);
    }

    /**
    * Set the list of index expressions.
    *
    * @param indices A list of expressions.
    * @throws NotAnOrphanException if an element of <b>indices</b> has a parent
    * object.
    */
    public void setIndices(List indices) {
        // clear out everything but the first item
        Expression name = (Expression)children.get(0);
        children.clear();
        children.add(name);
        for (Object o : indices) {
            addChild((Traversable)o);
        }
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
