package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.*;

/**
* <b>DoLoop</b> represents a C-style do-while loop.
*/
public class DoLoop extends Statement implements Loop, SymbolTable {

    /** The default print method */
    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = DoLoop.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** The internal look-up table */
    private Map<IDExpression, Declaration> symbol_table;

    /**
    * Constructs a new do loop with the given body statement and the condition
    * expression. Any non-compound statement body is normalized internally
    * to a compound statement.
    *
    * @param body the loop body statement.
    * @param condition the condition expression.
    * @throws IllegalArgumentException if <b>body</b> or <b>condition</b> is
    *       null.
    * @throws NotAnOrphanException if <b>body</b> or <b>condition</b> has a
    *       parent.
    */
    public DoLoop(Statement body, Expression condition) {
        super(2);
        object_print_method = class_print_method;
        symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
        if (!(body instanceof CompoundStatement)) {
            CompoundStatement cs = new CompoundStatement();
            cs.addStatement(body);
            body = cs;
        }
        addChild(body);
        addChild(condition);
        condition.setParens(false);
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
    * Prints a do loop to a print writer.
    *
    * @param l The loop to print.
    * @param o The writer on which to print the loop.
    */
    public static void defaultPrint(DoLoop l, PrintWriter o) {
        o.println("do");
        l.getBody().print(o);
        o.print("while");
        o.print("(");
        l.getCondition().print(o);
        o.println(");");
    }

    /* SymbolTable interface */
    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    /* Loop interface */
    public Statement getBody() {
        return (Statement)children.get(0);
    }

    /* Loop interface */
    public Expression getCondition() {
        return (Expression)children.get(1);
    }

    /* SymbolTable interface */
    public List<SymbolTable> getParentTables() {
        return SymbolTools.getParentTables(this);
    }

    /**
    * Returns the internal look-up table for symbols. This method is protected
    * for consistent symbol table management.
    *
    * @return the internal look-up table.
    */
    protected Map<IDExpression, Declaration> getTable() {
        return symbol_table;
    }

    /**
    * Sets the body statement with the given statement. The existing body
    * statement is overwritten. As in the constructor, any non-compound
    * statement body is normalized to a compound statement.
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
        setChild(0, body);
    }

    /**
    * Overrides the class print method, so that all subsequently
    * created objects will use the supplied method.
    *
    * @param m The new print method.
    */
    public static void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /**
    * Sets the condition expression with the given expression <b>cond</b>.
    *
    * @param cond the new condition expression.
    * @throws IllegalArgumentException if <b>cond</b> is null.
    * @throws NotAnOrphanException if <b>cond</b> has a parent object.
    */
    public void setCondition(Expression cond) {
        setChild(1, cond);
        cond.setParens(false);
    }

    /** Returns a clone of this do loop. */
    @Override
    public DoLoop clone() {
        DoLoop dl = (DoLoop)super.clone();
        // Creates its own symbol lookup table.
        dl.symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
        DFIterator<Declaration> iter =
                new DFIterator<Declaration>(dl, Declaration.class);
        iter.pruneOn(Declaration.class);
        iter.pruneOn(CompoundStatement.class);
        iter.reset(); // recheck with pruned classes.
        while (iter.hasNext()) {
            SymbolTools.addSymbols(dl, iter.next());
        }
        // Fixes obsolete symbol references in the IR.
        SymbolTools.relinkSymbols(dl);
        return dl;
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
