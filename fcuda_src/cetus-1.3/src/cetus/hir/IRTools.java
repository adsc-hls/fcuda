package cetus.hir;

import java.util.*;

/**
* <b>IRTools</b> provides tools that perform search/replace in the IR tree.
*/
public final class IRTools {

    private IRTools() {
    }

    /**
    * Searches the IR tree beginning at {@code t} for the Expression {@code e}.
    * 
    * @param t the traversable object to be searched.
    * @param e the expression to be searched for.
    * @return true if {@code t} contains {@code e} and false otherwise.
    */
    public static boolean containsExpression(Traversable t, Expression e) {
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            if (iter.next().equals(e)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Searches the IR tree beginning at {@code t} for any expressions in the
    * given collection of expressions.
    *
    * @param t the traversable object to be searched.
    * @param es the collection of expressions to be searched for.
    * @return true if {@code t} contains an expression in the collection
    * {@code es}.
    */
    public static boolean containsExpressions(
            Traversable t, Collection<? extends Expression> es) {
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            Expression child = iter.next();
            for (Expression e : es) {
                if (e.equals(child)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
    * Counts the number of times that the Expression {@code e} appears in
    * the IR tree {@code t}.
    *
    * @param t the traversable object to be searched.
    * @param e the expression to be counted.
    * @return the number of times {@code e} appears in {@code t}.
    */
    public static int countExpressions(Traversable t, Expression e) {
        int ret = 0;
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            if (iter.next().equals(e)) {
                ret++;
            }
        }
        return ret;
    }

    /**
    * Finds the first instance of the Expression {@code e} in the IR tree
    * {@code t}.
    *
    * @param t the traversable object to be searched.
    * @param e the expression to be searched for.
    * @return an expression from {@code t} that is equal to {@code e}.
    */
    public static Expression findExpression(Traversable t, Expression e) {
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            Expression child = iter.next();
            if (child.equals(e)) {
                return child;
            }
        }
        return null;
    }

    /**
    * Finds the list of instances of the Expression {@code e} in the IR tree
    * {@code t}. 
    *
    * @param t the traversable object to be searched.
    * @param e the expression to be searched for.
    * @return the list of matching expressions.
    */
    public static List<Expression>
            findExpressions(Traversable t, Expression e) {
        List<Expression> ret = new ArrayList<Expression>(4);
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            Expression child = iter.next();
            if (child.equals(e)) {
                ret.add(child);
            }
        }
        return ret;
    }

    /**
    * Checks the consistency of the IR subtree rooted at {@code t}.
    * It checks if every children of a parent node has a correct back link to
    * the parent node.
    *
    * @param t the traversable object to be checked.
    * @return true if it is consistent, false otherwise.
    */
    public static boolean checkConsistency(Traversable t) {
        DFIterator<Traversable> iter = new DFIterator<Traversable>(t);
        iter.next();
        while (iter.hasNext()) {
            Traversable tr = iter.next();
            Traversable p = tr.getParent();
            if (p == null ||
                Tools.identityIndexOf(p.getChildren(), tr) < 0) {
                PrintTools.printlnStatus(0, "Affected IR =", tr);
                PrintTools.printlnStatus(0, "Affected parent =", p);
                return false;
            }
        }
        return true;
    }

    /**
    * Replaces all instances of expression <var>x</var> on the IR tree
    * beneath <var>t</var> by <i>clones of</i> expression <var>y</var>.
    * Skips the immediate right hand side of member access expressions.
    *
    * @param t The location at which to start the search.
    * @param x The expression to be replaced.
    * @param y The expression to substitute.
    */
    public static void replaceAll(Traversable t, Expression x, Expression y) {
        List<Expression> matches = findExpressions(t, x);
        for (int i = 0; i < matches.size(); i++) {
            Expression match = matches.get(i);
            Traversable parent = match.getParent();
            if (parent instanceof AccessExpression &&
                    match == ((AccessExpression)parent).getRHS()) {
                /* don't replace these */
            } else {
                match.swapWith(y.clone());
            }
        }
        /*
        BreadthFirstIterator iter = new BreadthFirstIterator(t);
        for (;;) {
            Expression o = null;
            try {
                o = (Expression) iter.next(x.getClass());
            }
            catch(NoSuchElementException e) {
                break;
            }
            if (o.equals(x)) {
                if (o.getParent()instanceof AccessExpression
                        && ((AccessExpression) o.getParent()).getRHS() == o) {
                    // don't replace these
                } else {
                    if (o.getParent() == null) {
                        System.err.println("[ERROR] this " + o.toString() +
                                           " should be on the tree");
                    }
                    Expression copy = y.clone();
                    o.swapWith(copy);
                    if (copy.getParent() == null) {
                        System.err.println("[ERROR] " + y.toString() +
                                           " didn't get put on tree properly");
                    }
                }
            }
        }
        */
    }

    /**
    * Returns the Procedure to which the input traversable 
    * object belongs.
    *
    * @param t the traversable object to be searched from.
    * @return the procedure that contains {@code t}.
    */
    public static Procedure getParentProcedure(Traversable t) {
        while (t != null) {
            if (t instanceof Procedure) {
                break;
            }
            t = t.getParent();
        }
        return (Procedure)t;
    }

    /**
    * Returns the TranslationUnit to which the input traversable 
    * object belongs.
    *
    * @param t the traversable object to be searched from.
    * @return the translation unit that contains {@code t}.
    */
    public static TranslationUnit getParentTranslationUnit(Traversable t) {
        while (t != null) {
            if (t instanceof TranslationUnit) {
                break;
            }
            t = t.getParent();
        }
        return (TranslationUnit)t;
    }

    /**
    * Returns the last declaration that belongs to the given traverable object.
    *
    * @param t the traversable object to be searched.
    * @return the last declaration or null if not found.
    */
    public static Declaration getLastDeclaration(Traversable t) {
        DeclarationStatement decl_stmt = getLastDeclarationStatement(t);
        if (decl_stmt == null) {
            return null;
        } else {
            return decl_stmt.getDeclaration();
        }
    }

    /**
    * Returns the last declaration statement that belongs to the given
    * traverable object. 
    *
    * @param t the traversable object to be searched.
    * @return the last declaration statement or null if not found.
    */
    public static DeclarationStatement
            getLastDeclarationStatement(Traversable t) {
        DeclarationStatement decl_stmt = null;
        List<Traversable> children = t.getChildren();
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = children.get(i);
            if (child instanceof DeclarationStatement) {
                decl_stmt = (DeclarationStatement)child;
            } else if (!(child instanceof AnnotationStatement)) {
                break;
            }
        }
        return decl_stmt;
    }

    /**
    * Returns the first non-DeclarationStatement of the given traverable object.
    *
    * @param t the traversable object to be searched.
    * @return the first non-declaration statement or null if not found.
    */
    public static Statement getFirstNonDeclarationStatement(Traversable t) {
        Statement non_decl_stmt = null;
        List<Traversable> children = t.getChildren();
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = children.get(i);
            if (!(child instanceof DeclarationStatement) &&
                !(child instanceof AnnotationStatement)) {
                non_decl_stmt = (Statement)child;
                break;
            }
        }
        return non_decl_stmt;
    }

    /**
    * Replaces all occurrences of the specified variable {@code var} with the
    * given expression {@code expr} in the input expression {@code e}. If the
    * traversable object is the specified variable it returns a copy of the
    * expression {@code expr}.
    *
    * @param e the input expression.
    * @param var the variable to be replaced.
    * @param expr the expression to be substituted.
    * @return the cloned and modified expression. 
    */
    public static Expression
            replaceSymbol(Expression e, Symbol var, Expression expr) {
        if (var == null || var.getSymbolName().equals(e.toString()))
            return expr.clone();
        Expression ret = e.clone();
        Identifier id = new Identifier(var);
        List<Expression> matches = findExpressions(ret, id);
        for (int i = 0; i < matches.size(); i++) {
            Identifier match = (Identifier)matches.get(i);
            if (match.getSymbol() == var) {
                match.swapWith(expr.clone());
            }
        }
        return ret;
    }

    /**
    * Replaces all occurrences of the specified variable {@code var} with the
    * given expression {@code e} in the traversable object {@code t} in place.
    *
    * @param t the traversable object to be modified.
    * @param var the variable to be replaced.
    * @param e the expression to be substituted.
    */
    public static void
            replaceSymbolIn(Traversable t, Symbol var, Expression e) {
        if (t.getChildren() == null) {
            PrintTools.printlnStatus(0,
                    "[WARNING] no in-place replacement allowed in leaf nodes.");
            return;
        }
        // Processes traversable identifiers first.
        Identifier id = new Identifier(var);
        List<Expression> matches = findExpressions(t, id);
        for (int i = 0; i < matches.size(); i++) {
            Identifier match = (Identifier)matches.get(i);
            if (match.getSymbol() == var) {
                match.swapWith(e.clone());
            }
        }
        // Processes non-traversable identifiers within array specifiers.
        DFIterator<VariableDeclarator> iter =
                new DFIterator<VariableDeclarator>(t, VariableDeclarator.class);
        while (iter.hasNext()) {
            replaceSymbolInDeclarator(iter.next(), var, e);
        }
    }

    // This routine handles replacement of identifiers within array specifiers,
    // which are not traversable from the root (hence no symbol link). For this
    // reason, it will use the replaceAll() method which relies on the name of
    // the identifier.
    private static void replaceSymbolInDeclarator(
            VariableDeclarator dec, Symbol var, Expression e) {
        List array_specs = dec.getArraySpecifiers();
        if (array_specs.isEmpty()) {
            return;
        }
        Identifier old_id = new Identifier(var);
        ArraySpecifier array_spec = (ArraySpecifier)array_specs.get(0);
        for (int i = 0; i < array_spec.getNumDimensions(); i++) {
            Expression dim = array_spec.getDimension(i);
            if (dim == null) {
                continue;
            }
            if (dim.toString().equals(var.getSymbolName())) {
                array_spec.setDimension(i, e.clone());
            } else if (!(dim instanceof Identifier)) {
                replaceAll(dim, old_id, e.clone());
            }
        }
    }

    /**
    * Checks if the specified traversable object contains any identifier
    * derived from the given symbol {@code var}. This method only searches for
    * the derived identifier not the symbol itself.
    *  
    * @param t the traversable object to be searched.
    * @param var the symbol to be searched for.
    * @return true if there is any identifier derived from {@code var}, false
    * otherwise.
    */
    public static boolean containsSymbol(Traversable t, Symbol var) {
        boolean ret = false;
        if (t != null && var != null) {
            DFIterator<Identifier>iter =
                    new DFIterator<Identifier>(t, Identifier.class);
            while (iter.hasNext()) {
                if (iter.next().getSymbol() == var) {
                    ret = true;
                    break;
                }
            }
        }
        return ret;
    }

    /**
    * Checks if the specified traversable object contains any identifier
    * derived from the given set of symbols {@code vars}. This method only
    * searches for the derived identifier not the symbol itself.
    *
    * @param t the traversable object to be searched.
    * @param vars the set of symbols to be searched for.
    * @return true if there is any identifier derived from any symbol in
    * {@code vars}, false otherwise.
    */
    public static boolean containsSymbols(Traversable t, Set<Symbol> vars) {
        if (t == null) {
            return false;
        }
        DFIterator<Identifier> iter =
                new DFIterator<Identifier>(t, Identifier.class);
        while (iter.hasNext()) {
            if (vars.contains(iter.next().getSymbol())) {
                return true;
            }
        }
        return false;
    }

    /**
    * Simple check method for existence of intersection between the two given
    * sets of symbols.
    *
    * @param vars the first set of symbols.
    * @param symbols the second set of symbols.
    * @return true if the intersection is not empty, false otherwise.
    */
    public static boolean
            containsSymbols(Set<Symbol> vars, Set<Symbol> symbols) {
        for (Symbol symbol : symbols) {
            if (vars.contains(symbol)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Checks if the traversable object contains the specified type of object.
    *
    * @param t the traversable object to be searched.
    * @param type the class to be searched for.
    * @return true if {@code t} contains the type {@code type}.
    */
    public static boolean containsClass(
            Traversable t, Class<? extends Traversable> type) {
        if (t == null) {
            return false;
        }
        DFIterator<Traversable> iter = new DFIterator<Traversable>(t, type);
        return iter.hasNext();
    }

    /**
    * Checks if the traversable object contains the specified types of object.
    *
    * @param t the traversable object to be searched.
    * @param types the classes to be searched for.
    * @return true if {@code t} contains any of the set {@code types}.
    */
    public static boolean containsClasses(
            Traversable t, Set<Class<? extends Traversable>> types) {
        if (t == null) {
            return false;
        }
        for (Class<? extends Traversable> type : types) {
            if (containsClass(t, type)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Checks if the traversable object contains the specified type of binary
    * operations.
    *
    * @param t The traversable object being searched
    * @param op The binary operator being searched for
    * @return True if there is such an operation, False otherwise
    */
    public static boolean containsBinary(Traversable t, BinaryOperator op) {
        if (t == null) {
            return false;
        }
        DFIterator<BinaryExpression> iter =
                new DFIterator<BinaryExpression>(t, BinaryExpression.class);
        while (iter.hasNext()) {
            if (iter.next().getOperator() == op) {
                return true;
            }
        }
        return false;
    }

    /**
    * Checks if the traversable object contains the specified type of unary
    * operations.
    *
    * @param t The traversable object being searched
    * @param op The unary operator being searched for
    * @return True if there is such an operation, False otherwise
    */
    public static boolean containsUnary(Traversable t, UnaryOperator op) {
        if (t == null) {
            return false;
        }
        DFIterator<UnaryExpression> iter =
                new DFIterator<UnaryExpression>(t, UnaryExpression.class);
        while (iter.hasNext()) {
            if (iter.next().getOperator() == op) {
                return true;
            }
        }
        return false;
    }

    /**
    * Returns a list of unary expressions with the given unary operator.
    *
    * @param t the traversable object being searched.
    * @param op the unary operator being searched for.
    * @return the list of unary expressions.
    */
    public static List<UnaryExpression>
            getUnaryExpression(Traversable t, UnaryOperator op) {
        List<UnaryExpression> ret = new ArrayList<UnaryExpression>();
        DFIterator<UnaryExpression> iter =
                new DFIterator<UnaryExpression>(t, UnaryExpression.class);
        while (iter.hasNext()) {
            UnaryExpression ue = iter.next();
            if (ue.getOperator() == op) {
                ret.add(ue);
            }
        }
        return ret;
    }

    /**
    * Returns true if the traversable contains any side effects that change the
    * program state.
    *
    * @param t  The traversable object being searched
    * @return true if there is such a case, false otherwise.
    */
    public static boolean containsSideEffect(Traversable t) {
        if (t == null) {
            return false;
        }
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            Expression e = iter.next();
            if (e instanceof AssignmentExpression ||
                e instanceof FunctionCall ||
                e instanceof VaArgExpression ||
                e instanceof UnaryExpression && UnaryOperator.hasSideEffects(
                    ((UnaryExpression)e).getOperator())) {
                return true;
            }
        }
        return false;
    }

    /**
    * Returns the nearest ancestor object of the given traversable object
    * that has the specified type.
    *
    * @param t the traversable object to be searched from.
    * @param type the IR type being searched for.
    * @return the youngest ancestor of {@code t} having the type {@code type}.
    */
    @SuppressWarnings("unchecked")
    public static <T extends Traversable> T
            getAncestorOfType(Traversable t, Class<T> type) {
        if (t == null) {
            return null;
        }
        Traversable ret = t.getParent();
        while (ret != null && !type.isInstance(ret)) {
            ret = ret.getParent();
        }
        return (T)ret;
    }

    /**
    * Returns a list of descendents of the traversable object {@code t} with the
    * specified type {@code type}.
    *
    * @param t the traversable object to be searched.
    * @param type the IR type to be searched for.
    * @return the list of descendents having the type {@code type}.
    */
    public static <T extends Traversable> List<T>
            getDescendentsOfType(Traversable t, Class<T> type) {
        List<T> ret = (new DFIterator<T>(t, type)).getList();
        if (type.isInstance(t)) {
            ret.remove(0);
        }
        return ret;
    }

    /**
    * Checks if the specified traversable object {@code anc} is an ancestor of
    * the other traversable object {@code des} in the IR tree.
    * 
    * @param anc a possible ancestor of {@code des}.
    * @param des a possible descendant of {@code anc}.
    * @return true if {@code anc} is an ancestor of {@code des}, false if not.
    */
    public static boolean isAncestorOf(Traversable anc, Traversable des) {
        return isDescendantOf(des, anc);
    }

    /**
    * Checks if the specified traversable object {@code des} is a descendant of
    * the other traversable object {@code anc} in the IR tree.
    * 
    * @param des a possible descendant of {@code anc}.
    * @param anc a possible ancestor of {@code des}.
    * @return true if {@code des} is a descendant of {@code anc}, false if not. 
    */
    public static boolean isDescendantOf(Traversable des, Traversable anc) {
        Traversable t = des;
        while (t != null && t != anc) {
            t = t.getParent();
        }
        return (des != anc && t == anc);
    }

    /**
    * Returns true if there is a FunctionCall within the traversable.
    *
    * @param t  traversable object to be searched.
    * @return true if {@code t} contains a function call.
    */
    public static boolean containsFunctionCall(Traversable t) {
        return containsClass(t, FunctionCall.class);
    }

    /**
    * Returns a list of FunctionCall expressions within the traversable object.
    *
    * @param t the traversable object to be searched.
    * @return the list of function calls that appear in {@code t}.
    */
    public static List<FunctionCall> getFunctionCalls(Traversable t) {
        if (t == null) {
            return null;
        }
        return (new DFIterator<FunctionCall>(t, FunctionCall.class)).getList();
    }

    /**
    * Returns a list of pragma annotations that contain the specified string
    * keys and are attached to annotatable objects within the traversable object
    * {@code t}. For example, it can collect list of OpenMP pragmas having
    * a work-sharing directive {@code for} within a specific procedure.
    *
    * @param t the traversable object to be searched.
    * @param pragma_cls the type of pragmas to be searched for.
    * @param key the keyword to be searched for.
    * @return the list of matching pragma annotations.
    */
    public static <T extends PragmaAnnotation> List<T> collectPragmas(
            Traversable t, Class<T> pragma_cls, String key) {
        List<T> ret = new ArrayList<T>();
        DFIterator<Annotatable> iter =
                new DFIterator<Annotatable>(t, Annotatable.class);
        while (iter.hasNext()) {
            Annotatable at = iter.next();
            List<T> pragmas = at.getAnnotations(pragma_cls);
            for (int i = 0; i < pragmas.size(); i++) {
                T pragma = pragmas.get(i);
                if (pragma.containsKey(key)) {
                    ret.add(pragma);
                }
            }
        }
        return ret;
    }

    /**
    * Returns a list of the last Statements to be evaluated in the Procedure.
    * This method actually searches for return statements or the last statement
    * in traversal order if there is no return statement.
    */
    public static List<Statement> getLastStatements(Procedure proc) {
        List<Statement> ret = (new DFIterator<Statement>(
                proc, ReturnStatement.class)).getList();
        List<Traversable> children = proc.getBody().getChildren();
        try {
            Statement last_stmt = (Statement)children.get(children.size()-1);
            if (!ret.contains(last_stmt)) {
                ret.add(last_stmt);
            }
        } catch (IndexOutOfBoundsException ex) {
            Tools.exit("ERROR: function [" + proc.getName() +
                       "] is empty (needs at least one Statement)");
        }
        return ret;
        /* NOTE: Don't know if BFS is required: revert if there is any problem.
        List<Statement> last_stmt_list = new LinkedList<Statement>();
        BreadthFirstIterator iter = new BreadthFirstIterator(proc);
        // Case I: find all ReturnStatements
        for (;;) {
            ReturnStatement stmt = null;
            try {
                stmt = (ReturnStatement) iter.next(ReturnStatement.class);
                last_stmt_list.add(stmt);
            }
            catch(NoSuchElementException e) {
                break;
            }
        }
        // Case II: if no ReturnStatement is found, find the last Statement of the Procedure  
        List<Traversable> children = proc.getBody().getChildren();
        if (children.size() == 0)
            Tools.exit("ERROR: function [" + proc.getName() +
                       "] is empty (needs at least one Statement)");
        Statement last_stmt = (Statement)children.get(children.size() - 1);
        if (!last_stmt_list.contains(last_stmt)) {
            last_stmt_list.add(last_stmt);
        }
        return last_stmt_list;
        */
    }

    /**
    * Returns a list of statements having the specified type in the given
    * traversable object.
    *
    * @param t the traversable object to be examined.
    * @param type the type of statements to be collected.
    * @return the list of statements that are instances of {@code type}.
    */
    public static <T extends Statement> List<T>
            getStatementsOfType(Traversable t, Class<T> type) {
        return (new DFIterator<T>(t, type)).getList();
    }

    /**
    * Returns a list of expressions having the specified type in the given
    * traversable object.
    *
    * @param t the traversable object to be examined.
    * @param type the type of expressions to be collected.
    * @return the list of expressions that are instances of {@code type}.
    */
    public static <T extends Expression> List<T>
            getExpressionsOfType(Traversable t, Class<T> type) {
        return (new DFIterator<T>(t, type)).getList();
    }

    /**
    * Return a list of Procedures in a program.
    */
    public static List<Procedure> getProcedureList(Program program) {
        DFIterator<Procedure> iter =
                new DFIterator<Procedure>(program, Procedure.class);
        iter.pruneOn(Procedure.class);
        iter.pruneOn(Statement.class);
        iter.pruneOn(Declaration.class);
        return iter.getList();
    }

    /**
    * Removes all annotations with the specified type from the given traversable
    * object. 
    *
    * @param t the traversable object to be searched.
    * @param type the annotation types to be removed.
    */
    public static void removeAnnotations(
            Traversable t, Class<? extends Annotation> type) {
        DFIterator<Annotatable> iter =
                new DFIterator<Annotatable>(t, Annotatable.class);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(ExpressionStatement.class);
        while (iter.hasNext()) {
            iter.next().removeAnnotations(type);
        }
    }

    /**
    * Checks if the specified traversable object is located within an included
    * code section. This information could be useful when applying any
    * transformation to a code section since any changes on the code within an
    * included region is discarded when the header files are not expanded during
    * printing.
    *
    * @param t the traversable object to be examined.
    * @return true if {@code t} is part of the IR tree and within an included 
    * code section.
    */
    /*
    public static boolean isIncluded(Traversable t) {
        Traversable t1 = t;
        while (t1 != null
               && !((t1 = t1.getParent())instanceof TranslationUnit));
        if (t1 == null)
            return false;       // not part of IR tree
        // t1.getParent() should be translation unit
        TranslationUnit tu = (TranslationUnit) t1.getParent();
        int curr_pos = Tools.identityIndexOf(tu.getChildren(), t1);
        for (int i = curr_pos + 1; i < tu.getChildren().size(); i++) {
            Declaration decl = (Declaration) tu.getChildren().get(i);
            if (decl.
                containsAnnotation(PragmaAnnotation.class, "endinclude"))
                return true;
        }
        return false;
    }
    */
}
