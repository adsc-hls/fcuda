package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.*;

/**
* <b>ForLoop</b> represents a C-style for loop, typically having an initial
* statement, a condition expression, and a step expression.
*/
public class ForLoop extends Statement implements Loop, SymbolTable {

    /** The default print method for the loop */
    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ForLoop.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** The internal look-up table for symbols */
    private Map<IDExpression, Declaration> symbol_table;

    /**
    * Constructs a new for loop with the given initial expression <b>init</b>,
    * the condition expression <b>condition</b>, the step expression
    * <b>step</b>, and the body statement <b>body</b>. Any non-compound body
    * statement is internally normalized to a compound statement.
    *
    * @param init the initial statement.
    * @param condition the condition expression.
    * @param step the step expression.
    * @param body the body statement.
    * @throws NotAnOrphanException if <b>init</b>, <b>condition</b>,
    *       <b>step</b>, or <b>body</b> has a parent.
    */
    public ForLoop(Statement init, Expression condition, Expression step,
                   Statement body) {
        super(4);
        object_print_method = class_print_method;
        symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
        children.add(null);
        children.add(null);
        children.add(null);
        children.add(new CompoundStatement());
        if (init != null) {
            setChild(0, init);
        }
        if (condition != null) {
            setChild(1, condition);
            condition.setParens(false);
        }
        if (step != null) {
            setChild(2, step);
            step.setParens(false);
        }
        setBody(body);
    }

    /* SymbolTable interface */
    public void addDeclaration(Declaration decl) {
        throw new UnsupportedOperationException(
                "Add the declaration in the body compound statement instead.");
    }

    /* SymbolTable interface */
    public void addDeclarationBefore(Declaration ref, Declaration decl) {
        throw new UnsupportedOperationException(
                "Add the declaration in the body compound statement instead.");
    }

    /* SymbolTable interface */
    public void addDeclarationAfter(Declaration ref, Declaration decl) {
        throw new UnsupportedOperationException(
                "Add the declaration in the body compound statement instead.");
    }

    /**
    * Prints the for loop to the given print writer.
    *
    * @param l The loop to print.
    * @param o The writer on which to print the loop.
    */
    public static void defaultPrint(ForLoop l, PrintWriter o) {
        o.print("for (");
        if (l.getInitialStatement() != null) {
            l.getInitialStatement().print(o);
        } else {
            o.print(";");
        }
        o.print(" ");
        if (l.getCondition() != null) {
            l.getCondition().print(o);
        }
        o.print("; ");
        if (l.getStep() != null) {
            l.getStep().print(o);
        }
        o.println(")");
        l.getBody().print(o);
    }

    /* SymbolTable interface */
    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    /* Loop interface */
    public Statement getBody() {
        return (Statement)children.get(3);
    }

    /* Loop interface */
    public Expression getCondition() {
        return (Expression)children.get(1);
    }

    /**
    * Returns the initial statement of the for loop allowing null.
    *
    * @return the initial statement.
    */
    public Statement getInitialStatement() {
        return (Statement)children.get(0);
    }

    /* SymbolTable interface */
    public List<SymbolTable> getParentTables() {
        return SymbolTools.getParentTables(this);
    }

    /**
    * Returns the step expression of the for loop allowing null.
    *
    * @return the step expression.
    */
    public Expression getStep() {
        return (Expression)children.get(2);
    }

    /**
    * Returns the internal look-up table. This method is protected for
    * consistent management of symbols.
    *
    * @return the look-up table.
    */
    protected Map<IDExpression, Declaration> getTable() {
        return symbol_table;
    }

    /**
    * Sets the body statement of the loop with the given statement <b>body</b>.
    * Any non-compound body statement is normalized to a compound statement.
    * 
    * @param body the new body statement.
    * @throws NotAnOrphanException if <b>body</b> has a parent.
    */
    public void setBody(Statement body) {
        if (!(body instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            if (body != null) {
                cs.addStatement(body);
            }
            body = cs;
        }
        setChild(3, body);
    }

    /**
    * Overrides the class print method, so that all subsequently created
    * objects will use the supplied method.
    *
    * @param m The new print method.
    */
    static public void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /**
    * Sets the condition expression with the given expression <b>cond</b>.
    *
    * @param cond the new condition expression (null is allowed).
    * @throws NotAnOrphanException if <b>cond</b> has a parent object.
    */
    public void setCondition(Expression cond) {
        if (cond != null && cond.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (getCondition() != null) {
            getCondition().setParent(null);
        }
        children.set(1, cond);
        if (cond != null) {
            cond.setParent(this);
            cond.setParens(false);
        }
    }

    /**
    * Sets the initial statement with the given statement <b>stmt</b>.
    *
    * @param stmt the new initial statement (null is allowed).
    * @throws NotAnOrphanException if <b>stmt</b> has a parent object.
    */
    public void setInitialStatement(Statement stmt) {
        if (stmt != null && stmt.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (getInitialStatement() != null) {
            getInitialStatement().setParent(null);
        }
        children.set(0, stmt);
        if (stmt != null) {
            stmt.setParent(this);
            if (stmt instanceof DeclarationStatement) {
                Declaration decl =
                        ((DeclarationStatement)stmt).getDeclaration();
                SymbolTools.addSymbols(this, decl);
            }
        }
    }

    /**
    * Sets the step expression with the given expression <b>step</b>.
    *
    * @param step the new step expression (null is allowed).
    * @throws NotAnOrphanException if <b>step</b> has a parent object.
    */
    public void setStep(Expression step) {
        if (step != null && step.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (getStep() != null) {
            getStep().setParent(null);
        }
        children.set(2, step);
        if (step != null) {
            step.setParent(this);
            step.setParens(false);
        }
    }

    /**
    * Returns a clone of this for loop.
    *
    * @return the cloned for loop.
    */
    @Override public ForLoop clone() {
        ForLoop fl = (ForLoop)super.clone();
        // Builds the internal look-up table.
        // There is no need for building an internal look-up table for the loop
        // in a C program (declarations are within the compound statement) but
        // it doesn't harm to make the clone operation consistent.
        fl.symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
        DFIterator<Declaration> iter =
                new DFIterator<Declaration>(fl, Declaration.class);
        iter.pruneOn(Declaration.class);
        iter.pruneOn(CompoundStatement.class);
        iter.reset();
        while (iter.hasNext()) {
            SymbolTools.addSymbols(fl, iter.next());
        }
        // Fixes obsolete symbol references in the IR.
        SymbolTools.relinkSymbols(fl);
        return fl;
    }

    /* SymbolTable interface */
    public Set<Symbol> getSymbols() {
        return SymbolTools.getSymbols(this);
    }

    /* SymbolTable interface */
    public Set<Declaration> getDeclarations() {
        return new LinkedHashSet<Declaration>(symbol_table.values());
    }

    /* SymbolTable interface */
    public boolean containsSymbol(Symbol symbol) {
        for (IDExpression id : symbol_table.keySet()) {
            if (id instanceof Identifier &&
                symbol.equals(((Identifier) id).getSymbol())) {
                return true;
            }
        }
        return false;
    }

    /* SymbolTable interface */
    public boolean containsDeclaration(Declaration decl) {
        return symbol_table.containsValue(decl);
    }

}
