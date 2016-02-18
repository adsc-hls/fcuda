package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* Represents a if statement in C programs.
*/
public class IfStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = IfStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Create an <var>if</var> statement that has no <var>else</var> clause.
    * Any <var>true_clause</var> that is not a compound statement is normalized
    * to a compound statement.
    *
    * @param condition The condition tested by the statement.
    * @param true_clause The code to execute if the condition is true.
    * @throws IllegalArgumentException if <b>condition</b> or <b>true_clause</b>
    * is null.
    * @throws NotAnOrphanException if <b>condition</b> or <b>true_clause</b> has
    * a parent.
    */
    public IfStatement(Expression condition, Statement true_clause) {
        super(2);
        object_print_method = class_print_method;
        addChild(condition);
        if (!(true_clause instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            cs.addStatement(true_clause);
            true_clause = cs;
        }
        addChild(true_clause);
    }

    /**
    * Create an <var>if</var> statement that has an <var>else</var> clause.
    * <var>true_clause</var> and <var>false_clause</var> are normalized to
    * compound statements.
    *
    * @param condition The condition tested by the statement.
    * @param true_clause The code to execute if the condition is true.
    * @param false_clause The code to execute if the condition is false.
    * @throws IllegalArgumentException if <b>condition</b>, <b>true_clause</b>,
    * or <b>false_clause</b> is null.
    * @throws NotAnOrphanException if <b>condition</b>, <b>true_clause</b>, or
    * <b>false_clause</b> has a parent.
    */
    public IfStatement(Expression condition, Statement true_clause,
                       Statement false_clause) {
        super(3);
        object_print_method = class_print_method;
        addChild(condition);
        if (!(true_clause instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            cs.addStatement(true_clause);
            true_clause = cs;
        }
        addChild(true_clause);
        if (!(false_clause instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            cs.addStatement(false_clause);
            false_clause = cs;
        }
        addChild(false_clause);
    }

    /**
    * Prints an if statement to a stream.
    *
    * @param s The statement to print.
    * @param o The wirter on which to print the statement.
    */
    public static void defaultPrint(IfStatement s, PrintWriter o) {
        o.print("if (");
        s.getControlExpression().print(o);
        o.println(")");
        s.getThenStatement().print(o);
        if (s.getElseStatement() != null) {
            o.println("\nelse");
            s.getElseStatement().print(o);
        }
    }

    /** Returns the expression used as a branch condition. */
    public Expression getControlExpression() {
        return (Expression)children.get(0);
    }

    /**
    * Sets the condition expression with the specified new condition.
    *
    * @param cond the new condition expression.
    * @throws IllegalArgumentException if <b>cond</b> is null.
    * @throws NotAnOrphanException if <b>cond</b> has a parent.
    */
    public void setControlExpression(Expression cond) {
        setChild(0, cond);
    }

    /** Returns the then clause of the if statement. */
    public Statement getThenStatement() {
        return (Statement)children.get(1);
    }

    /**
    * Sets the then clause with the specified new statement. <b>stmt</b> is
    * normalized to a compound statement.
    *
    * @param stmt the new statement for the then clause.
    * @throws IllegalArgumentException if <b>stmt</b> is null.
    * @throws NotAnOrphanException if <b>stmt</b> has a parent.
    */
    public void setThenStatement(Statement stmt) {
        if (!(stmt instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            cs.addStatement(stmt);
            stmt = cs;
        }
        setChild(1, stmt);
    }

    /**
    * Returns the false clause of the if statement.
    *
    * @return the false clause or null (if it does not exist).
    */
    public Statement getElseStatement() {
        if (children.size() > 2) {
            return (Statement)children.get(2);
        } else {
            return null;
        }
    }

    /**
    * Sets the else clause with the specified new statement. <b>stmt</b> is
    * normalized to a compound statement.
    *
    * @param stmt the new statement for the else clause.
    * @throws IllegalArgumentException if <b>stmt</b> is null.
    * @throws NotAnOrphanException if <b>stmt</b> has a parent.
    */
    public void setElseStatement(Statement stmt) {
        if (getElseStatement() != null) {
            setChild(2, stmt);
        } else {
            addChild(stmt);
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

    /** Returns a clone of the if statement. */
    @Override
    public IfStatement clone() {
        IfStatement is = (IfStatement)super.clone();
        return is;
    }

}
