package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.*;

/**
* <b>WhileLoop</b> represents a while loop having a condition expression and
* a loop body.
*/
public class WhileLoop extends Statement implements Loop, SymbolTable {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = WhileLoop.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** Internal look-up table for symbols. */
    private Map<IDExpression, Declaration> symbol_table;

    /**
    * Constructs a new while loop with the given condition expression
    * <b>condition</b> and the loop body <b>body</b>. Any non-compound statement
    * body is normalized to a compound statement.
    *
    * @param condition the condition expression.
    * @param body the body statement.
    * @throws IllegalArgumentException if <b>condition</b> or <b>body</b> is
    *   null.
    * @throws NotAnOrphanException if <b>condition</b> or <b>body</b> has a
    *   parent.
    */
    public WhileLoop(Expression condition, Statement body) {
        super(2);
        object_print_method = class_print_method;
        symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
        addChild(condition);
        condition.setParens(false);
        if (!(body instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            cs.addStatement(body);
            body = cs;
        }
        addChild(body);
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
    * Prints a loop to a stream.
    *
    * @param l The loop to print.
    * @param o The writer on which to print the loop.
    */
    public static void defaultPrint(WhileLoop l, PrintWriter o) {
        o.print("while (");
        l.getCondition().print(o);
        o.println(")");
        l.getBody().print(o);
    }

    /* SymbolTable interface */
    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    /* Loop interface */
    public Statement getBody() {
        return (Statement)children.get(1);
    }

    /**
    * Sets the body statement with the given statement <b>body</b>. Any
    * non-compound statement is normalized to a compound statement.
    *
    * @param body the new body statement.
    * @throws IllegalArgumentException if <b>body</b> is null.
    * @throws NotAnOrphanException if <b>body</b> has a parent object.
    */
    public void setBody(Statement body) {
        if (!(body instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            cs.addStatement(body);
            body = cs;
        }
        setChild(1, body);
    }

    /* Loop interface */
    public Expression getCondition() {
        return (Expression)children.get(0);
    }

    /**
    * Sets the condition expression with the given expression <b>cond</b>.
    *
    * @param cond the new condition expression.
    * @throws IllegalArgumentException if <b>cond</b> is null.
    * @throws NotAnOrphanException if <b>cond</b> has a parent object.
    */
    public void setCondition(Expression cond) {
        setChild(0, cond);
        cond.setParens(false);
    }

    /* SymbolTable interface */
    public List<SymbolTable> getParentTables() {
        return SymbolTools.getParentTables(this);
    }

    /**
    * Returns the internal look-up table for symbols. This method is protected.
    *
    * @return the internal look-up table.
    */
    protected Map<IDExpression, Declaration> getTable() {
        return symbol_table;
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

    /** Returns a clone of this while loop. */
    @Override
    public WhileLoop clone() {
        WhileLoop wl = (WhileLoop)super.clone();
        wl.symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
        DFIterator<Declaration> iter =
                new DFIterator<Declaration>(wl, Declaration.class);
        iter.pruneOn(Declaration.class);
        iter.pruneOn(CompoundStatement.class);
        iter.reset();
        while (iter.hasNext()) {
            SymbolTools.addSymbols(wl, iter.next());
        }
        // Fixes obsolete symbol references in the IR.
        SymbolTools.relinkSymbols(wl);
        return wl;
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
                symbol.equals(((Identifier)id).getSymbol())) {
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
