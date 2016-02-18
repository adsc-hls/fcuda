package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents an expression containing a compound statement - GCC extension.
*/
public class StatementExpression extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = StatementExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Constructs a statement expression with the specified compound statement.
    */
    public StatementExpression(CompoundStatement stmt) {
        object_print_method = class_print_method;
        // addChild is not used since a statement is inserted.
        children.add(stmt);
        stmt.setParent(this);
    }

    /** Prints the statement expression on the specified print writer */
    public static void defaultPrint(StatementExpression e, PrintWriter o) {
        o.print("(");
        e.getStatement().print(o);
        o.print(")");
    }

    /** Returns the child statement of the statement expression */
    public CompoundStatement getStatement() {
        return (CompoundStatement)children.get(0);
    }

}
