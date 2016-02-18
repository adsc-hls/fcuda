package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.NoSuchElementException;

/**
* Represents a goto statement having a target label.
*/
public class GotoStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = GotoStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create a new goto statement.
    */
    public GotoStatement() {
        object_print_method = class_print_method;
    }

    /**
    * Create a new goto statement with the specified label.
    *
    * @param label the target label of the goto statement.
    * @throws IllegalArgumentException if <b>label</b> is null.
    * @throws NotAnOrphanException if <b>label</b> has a parent.
    */
    public GotoStatement(Expression label) {
        object_print_method = class_print_method;
        addChild(label);
    }

    /**
    * Returns the label name in the goto statement.
    */
    public Expression getExpression() {
        return (Expression)children.get(0);
    }

    /**
    * Sets the target label name with the specified expression.
    * 
    * @throws IllegalArgumentException if <b>label</b> is null.
    * @throws NotAnOrphanException if <b>label</b> has a parent.
    */
    public void setExpression(Expression label) {
        setChild(0, label);
    }

    /**
    * Prints a case label to a stream.
    *
    * @param s The goto statement to print.
    * @param o The writer on which to print the goto statement.
    */
    public static void defaultPrint(GotoStatement s, PrintWriter o) {
        o.print("goto ");
        s.getExpression().print(o);
        o.print(";");
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
    * Returns the {@link Label} object that this GotoStatement Jumps to.
    *
    * @return the Label.
    */
    public Label getTarget() {
        DFIterator<Label> iter =
                new DFIterator<Label>(getProcedure(), Label.class);
        while (iter.hasNext()) {
            Label label = iter.next();
            if (label.getName().equals(getExpression())) {
                return label;
            }
        }
        return null;
    }

}
