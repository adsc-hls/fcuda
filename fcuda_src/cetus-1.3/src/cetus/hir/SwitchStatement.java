package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a switch statement in C programs.
*/
public class SwitchStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = SwitchStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Creates a new switch statement with the specified expression.
    *
    * @param value the expression to be evaluated for branching.
    * @throws IllegalArgumentException if <b>value</b> is null.
    * @throws NotAnOrphanException if <b>value</b> has a parent.
    */
    public SwitchStatement(Expression value) {
        super(2);
        object_print_method = class_print_method;
        addChild(value);
        addChild(new CompoundStatement());
    }

    /**
    * Constructs a new switch statement with the specified value expression and
    * the body statement.
    *
    * @param value the expression to be evaluated for branching.
    * @param stmt the body of the switch statement.
    * @throws IllegalArgumentException if <b>value</b> or <b>stmt</b> is null.
    * @throws NotAnOrphanException if <b>value</b> or <b>stmt</b> has a parent.
    */
    public SwitchStatement(Expression value, CompoundStatement stmt) {
        super(2);
        object_print_method = class_print_method;
        addChild(value);
        addChild(stmt);
    }

    /**
    * Prints a switch statement to a stream.
    *
    * @param s The statement to print.
    * @param o The writer on which to print the statement.
    */
    public static void defaultPrint(SwitchStatement s, PrintWriter o) {
        o.print("switch (");
        s.getExpression().print(o);
        o.println(")");
        s.getBody().print(o);
    }

    /** Returns the body statement of the switch statement. */
    public CompoundStatement getBody() {
        return (CompoundStatement)children.get(1);
    }

    /** Returns the value expression of the switch statement. */
    public Expression getExpression() {
        return (Expression)children.get(0);
    }

    /**
    * Sets the body of the switch statement with the specified new statement.
    *
    * @param stmt the new body statement.
    * @throws IllegalArgumentException if <b>stmt</b> is null.
    * @throws NotAnOrphanException if <b>stmt</b> has a parent.
    */
    public void setBody(CompoundStatement stmt) {
        setChild(1, stmt);
    }

    /**
    * Sets the value of the switch statement with the specified new value.
    *
    * @param value the new value expression.
    * @throws IllegalArgumentException if <b>value</b> is null.
    * @throws NotAnOrphanException if <b>value</b> has a parent.
    */
    public void setExpression(Expression value) {
        setChild(0, value);
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

    /** Returns a clone of the switch statement. */
    @Override
    public SwitchStatement clone() {
        SwitchStatement ss = (SwitchStatement)super.clone();
        return ss;
    }

}
