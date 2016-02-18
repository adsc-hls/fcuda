package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
 * Provides tools for querying information related to For Loop objects
 */
public class LoopTools {

    // Flag for checking if all loops are named.
    private static boolean is_loop_named = false;

    /**
     * Constructor - not used at all.
     */
    private LoopTools() {
    }

    /*
     * Use the following static functions only with **ForLoops** that are
     * identified
     * as **CANONICAL** using **isCanonical**
     * getIncrementExpression(loop)
     * getIndexVariable(loop)
     * getLowerBoundExpression(loop)
     * getUpperBoundExpression(loop)
     */
    /**
     * Get the expression that represents the actual increment value for the
     * loop. This might be an integer constant or a symbolic value.
     */
    public static Expression getIncrementExpression(Loop loop) {
        Expression loopInc = null;
        if (loop instanceof ForLoop) {
            ForLoop for_loop = (ForLoop)loop;
            // determine the step
            Expression step_expr = for_loop.getStep();
            if (step_expr instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression)step_expr;
                AssignmentOperator aop = ae.getOperator();
                Expression rhs = Symbolic.simplify(ae.getRHS());
                if (aop.equals(AssignmentOperator.NORMAL) &&
                    rhs instanceof BinaryExpression) {
                    BinaryExpression brhs = (BinaryExpression)rhs;
                    BinaryOperator bop = brhs.getOperator();
                    if (bop.equals(BinaryOperator.ADD)) {
                        loopInc = brhs.getLHS();
                    } else if (bop.equals(BinaryOperator.SUBTRACT)) {
                        loopInc = Symbolic.multiply(
                                new IntegerLiteral(-1), brhs.getLHS());
                    }
                } else if (aop.equals(AssignmentOperator.ADD)) {
                    loopInc = rhs;
                } else if (aop.equals(AssignmentOperator.SUBTRACT)) {
                    loopInc = Symbolic.multiply(new IntegerLiteral(-1), rhs);
                }
            } else if (step_expr instanceof UnaryExpression) {
                UnaryExpression uexpr = (UnaryExpression)step_expr;
                UnaryOperator op = uexpr.getOperator();
                if (op == UnaryOperator.PRE_INCREMENT ||
                    op == UnaryOperator.POST_INCREMENT) {
                    loopInc = new IntegerLiteral(1);
                } else {
                    loopInc = new IntegerLiteral(-1);
                }
            }
        }
        if (!(loopInc instanceof IntegerLiteral)) {
            RangeDomain rd = RangeAnalysis.query((Statement)loop);
            loopInc = rd.substituteForward(loopInc);
        }
        return loopInc;
    }

    /**
     * Get loop increment expression and check if it is an integer constant
     */
    public static boolean isIncrementConstant(Loop loop) {
        Expression inc = getIncrementExpression(loop);
        if (inc instanceof IntegerLiteral) {
            return true;
        }
        return false;
    }

    /**
     * Get loop index variable, if loop is canonical
     */
    public static Expression getIndexVariable(Loop loop) {
        Expression indexVar = null;
        // Handle for loops here
        if (loop instanceof ForLoop) {
            // determine the name of the index variable
            ForLoop for_loop = (ForLoop)loop;
            Expression step_expr = for_loop.getStep();
            if (step_expr instanceof AssignmentExpression) {
                indexVar = ((AssignmentExpression)step_expr).getLHS().clone();
            } else if (step_expr instanceof UnaryExpression) {
                UnaryExpression uexpr = (UnaryExpression)step_expr;
                indexVar = uexpr.getExpression().clone();
            }
        }
        // Handle other loop types
        else {
        }
        return indexVar;
    }

    /**
     * Returns a simplified lower bound expression for the loop 
     */
    public static Expression getLowerBoundExpression(Loop loop) {
        Expression lb = null;
        if (loop instanceof ForLoop) {
            ForLoop for_loop = (ForLoop)loop;
            // determine lower bound for index variable of this loop
            Statement stmt = for_loop.getInitialStatement();
            if (stmt instanceof ExpressionStatement) {
                Expression rhs = ((AssignmentExpression)
                                 ((ExpressionStatement)stmt).
                                 getExpression()).getRHS();
                lb = Symbolic.simplify(rhs);
            } else if (stmt instanceof DeclarationStatement) {  // Error
            }
        }
        if (!(lb instanceof IntegerLiteral)) {
            RangeDomain rd = RangeAnalysis.query((Statement)loop);
            lb = rd.substituteForward(lb);
        }
        return lb;
    }

    /**
     * Check if the lower bound expression is an integer constant
     * @param loop
     * @return true if it is, false otherwise.
     */
    public static boolean isLowerBoundConstant(Loop loop) {
        Expression lb;
        lb = getLowerBoundExpression(loop);
        if (lb instanceof IntegerLiteral) {
            return true;
        }
        return false;
    }

    /**
     * Returns a simplified upper bound expression for the loop
     * @param loop
     * @return the upper bound of the loop.
     */
    public static Expression getUpperBoundExpression(Loop loop) {
        Expression ub = null;
        if (loop instanceof ForLoop) {
            ForLoop for_loop = (ForLoop)loop;
            // determine upper bound for index variable of this loop
            BinaryExpression cond_expr =
                    (BinaryExpression)for_loop.getCondition();
            Expression rhs = cond_expr.getRHS();
            Expression step_size = getIncrementExpression(loop);
            BinaryOperator cond_op = cond_expr.getOperator();
            if (cond_op.equals(BinaryOperator.COMPARE_LT)) {
                ub = Symbolic.subtract(rhs, step_size);
            } else if ((cond_op.equals(BinaryOperator.COMPARE_LE)) ||
                       (cond_op.equals(BinaryOperator.COMPARE_GE))) {
                ub = Symbolic.simplify(rhs);
            } else if (cond_op.equals(BinaryOperator.COMPARE_GT)) {
                ub = Symbolic.add(rhs, step_size);
            }
        }
        if (!(ub instanceof IntegerLiteral)) {
            RangeDomain rd = RangeAnalysis.query((Statement)loop);
            ub = rd.substituteForward(ub);
        }
        return ub;
    }

    /**
     * Check if loop upper bound is an integer constant
     */
    public static boolean isUpperBoundConstant(Loop loop) {
        Expression ub;
        ub = getUpperBoundExpression(loop);
        if (ub instanceof IntegerLiteral) {
            return true;
        }
        return false;
    }

    /**
     * Calculate the loop nest of this loop
     */
    public static LinkedList<Loop> calculateLoopNest(Loop loop) {
        LinkedList<Loop> loopNest = new LinkedList<Loop>();
        loopNest.add(loop);
        Traversable t = loop.getParent();
        while (t != null) {
            if (t instanceof ForLoop) {
                loopNest.addFirst((Loop)t);
            }
            t = t.getParent();
        }
        return loopNest;
    }

    /**
     * Get common enclosing loops for loop1 and loop2
     */
    public static LinkedList<Loop> getCommonNest(Loop loop1, Loop loop2) {
        LinkedList<Loop> commonNest = new LinkedList<Loop>();
        List<Loop> nest1 = calculateLoopNest(loop1);
        List<Loop> nest2 = calculateLoopNest(loop2);
        for (Loop l1 : nest1) {
            if (nest2.contains(l1)) {
                commonNest.add(l1);
            }
        }
        return commonNest;
    }

    /**
     * Check if loop is canonical, FORTRAN DO Loop format
     */
    /*
     * Following checks are performed:
     * - Initial statement contains assignment expression for loop index
     * - Simple conditional expression
     * - index variable increment with positive stride
     * - check if index variable is invariant within loop body
     */
    public static boolean isCanonical(Loop loop) {
        if (loop instanceof ForLoop) {
            ForLoop forloop = (ForLoop)loop;
            Identifier index_variable = null;
            //check initial statement and obtain index variable for loop
            index_variable = isInitialAssignmentExpression((Loop)forloop);
            if (index_variable == null) {
                return false;
            }
            //check loop condition based on index variable obtained
            if (checkLoopCondition((Loop)forloop, index_variable) == null) {
                return false;
            }
            //check loop step
            if ((checkIncrementExpression((Loop)forloop, index_variable)) == 0){
                return false;
            }
            //check index invariant
            if ((isIndexInvariant((Loop)forloop, index_variable)) == false) {
                return false;
            }
        } else if (loop instanceof WhileLoop) {
            return false;
        } else if (loop instanceof DoLoop) {
            return false;
        }
        // in the future it should handle other loops
        return true;
    }

    static private Identifier isInitialAssignmentExpression(Loop loop) {
        Statement init_stmt = ((ForLoop)loop).getInitialStatement();
        if (init_stmt instanceof ExpressionStatement) {
            Expression exp = ((ExpressionStatement)init_stmt).getExpression();
            if (exp instanceof AssignmentExpression) {
                AssignmentExpression assignment_exp = (AssignmentExpression)exp;
                AssignmentOperator op = assignment_exp.getOperator();
                if (op.equals(AssignmentOperator.NORMAL)) {
                    Expression lhs = Symbolic.simplify(assignment_exp.getLHS());
                    Expression rhs = Symbolic.simplify(assignment_exp.getRHS());
                    if (lhs instanceof Identifier) {
                        return ((Identifier)lhs);
                    }
                }
            }
        }
        return null;
    }

    static private Expression
            checkLoopCondition(Loop loop, Identifier induction_variable) {
        Expression loopbound = null;
        Expression cond_exp = ((ForLoop)loop).getCondition();
        if (cond_exp instanceof BinaryExpression) {
            BinaryExpression bin_condexp = (BinaryExpression)cond_exp;
            BinaryOperator bop = bin_condexp.getOperator();
            Expression lhs = Symbolic.simplify(bin_condexp.getLHS());
            Expression rhs = Symbolic.simplify(bin_condexp.getRHS());
            //if ((bop.equals(BinaryOperator.COMPARE_LT)) ||
            //    (bop.equals(BinaryOperator.COMPARE_LE)))
            if ((bop.equals(BinaryOperator.COMPARE_LT)) ||
                (bop.equals(BinaryOperator.COMPARE_LE)) ||
                (bop.equals(BinaryOperator.COMPARE_GT)) ||
                (bop.equals(BinaryOperator.COMPARE_GE))) {
                if (lhs.equals((Expression)induction_variable)) {
                    loopbound = rhs.clone();
                }
            }
        }
        return loopbound;
    }

    static private int checkIncrementExpression(Loop loop, Identifier id) {
        int increasing = 1;
        int decreasing = -1;
        int indeterminate = 0;
        Expression exp = ((ForLoop)loop).getStep();
        if (exp instanceof UnaryExpression) {
            UnaryExpression unary_exp = (UnaryExpression)exp;
            UnaryOperator uop = unary_exp.getOperator();
            Expression child = unary_exp.getExpression();
            if ((uop.equals(UnaryOperator.POST_INCREMENT) ||
                uop.equals(UnaryOperator.PRE_INCREMENT)) &&
                child.equals(id)) {
                return increasing;
            } else if ((uop.equals(UnaryOperator.POST_DECREMENT) ||
                        uop.equals(UnaryOperator.PRE_DECREMENT)) &&
                        child.equals(id)) {
                return decreasing;
            } else {
                return indeterminate;
            }
        } else if (exp instanceof AssignmentExpression) {
            AssignmentExpression assign_exp = (AssignmentExpression)exp;
            AssignmentOperator aop = assign_exp.getOperator();
            Expression alhs = assign_exp.getLHS(), arhs = assign_exp.getRHS();
            if (!alhs.equals(id)) {
                return indeterminate;
            }
            if (aop.equals(AssignmentOperator.NORMAL)) {
                if (arhs instanceof BinaryExpression) {
                    BinaryExpression bin_exp = (BinaryExpression)arhs;
                    BinaryOperator bop = bin_exp.getOperator();
                    if (bop.equals(BinaryOperator.ADD)) {
                        // Simplify the LHS and RHS of the binary expression to 
                        // accurately state whether we have a canonical
                        // increment expression or not
                        Expression rhs = Symbolic.simplify(bin_exp.getRHS());
                        Expression lhs = Symbolic.simplify(bin_exp.getLHS());
                        if (lhs.equals(id)) {
                            return increasing;
                        } else if (rhs.equals(id)) {
                            return increasing;
                        } else {
                            return indeterminate;
                        }
                    } else if (bop.equals(BinaryOperator.SUBTRACT)) {
                        // Simplify the LHS and RHS of the binary expression to 
                        // accurately state whether we have a canonical
                        // increment expression or not
                        Expression rhs = Symbolic.simplify(bin_exp.getRHS());
                        Expression lhs = Symbolic.simplify(bin_exp.getLHS());
                        if (lhs.equals(id)) {
                            return decreasing;
                        } else if (rhs.equals(id)) {
                            return decreasing;
                        } else {
                            return indeterminate;
                        }
                    }
                }
            } else if (aop.equals(AssignmentOperator.ADD)) {
                Expression lhs = Symbolic.simplify(alhs);
                if (lhs.equals(id)) {
                    return increasing;
                }
            } else if (aop.equals(AssignmentOperator.SUBTRACT)) {
                Expression lhs = Symbolic.simplify(alhs);
                if (lhs.equals(id)) {
                    return decreasing;
                }
            }
        }
        return indeterminate;
    }

    /**
     * Checks if loop body contains a function call
     */
    public static boolean containsFunctionCall(Loop loop) {
        if (loop instanceof ForLoop) {
            return (IRTools.containsClass(loop.getBody(), FunctionCall.class));
        } else {
            return true;
        }
    }

    /**
     * Check if the loop contains a function call that can be
     * tested for data dependences/that can be eventually parallelized
     */
    public static boolean containsOnlyParallelizableCall(Loop loop) {
        DFIterator<FunctionCall> iter =
                new DFIterator<FunctionCall>(loop, FunctionCall.class);
        while (iter.hasNext()) {
            FunctionCall fc = iter.next();
            if (!StandardLibrary.isSideEffectFree(fc)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if this loop and inner loops form a perfect nest
     */
    public static boolean isPerfectNest(Loop loop) {
        boolean pnest = false;
        List children;
        Object o = null;
        Statement stmt = loop.getBody();
        FlatIterator<Traversable> iter = new FlatIterator<Traversable>(stmt);
        if (iter.hasNext()) {
            boolean skip = false;
            do {
                o = (Statement)iter.next(Statement.class);
                if (o instanceof AnnotationStatement) {
                    skip = true;
                } else {
                    skip = false;
                }
            } while ((skip) && (iter.hasNext()));
            if (o instanceof ForLoop) {
                pnest = (isPerfectNest((Loop)o));
                // The ForLoop contains additional statements after the end
                // of the first ForLoop. This is interpreted as
                // a non-perfect nest for dependence testing
                if (iter.hasNext()) {
                    pnest = false;
                }
            } else if (o instanceof CompoundStatement) {
                children = ((Statement)o).getChildren();
                Statement s = (Statement)children.get(0);
                if (s instanceof ForLoop)
                    pnest = (isPerfectNest((Loop)s));
                else
                    pnest = false;
            } else if (containsLoop(loop)) {
                PrintTools.println("Loop is not perfectly nested", 8);
                pnest = false;
            } else {
                PrintTools.println("Loop is perfectly nested", 8);
                pnest = true;
            }
        }
        return pnest;
    }

    /**
     * Check if loop body contains another loop
     */
    public static boolean containsLoop(Loop loop) {
        // Test whether a ForLoop contains another ForLoop
        if (loop instanceof ForLoop) {
            return (IRTools.containsClass(loop.getBody(), ForLoop.class));
        } else {
            return true;
        }
    }

    /**
     * Check if the index variable is defined within the loop body
     */
    public static boolean isIndexInvariant(Loop loop, Identifier id) {
        Set<Symbol> def_symbol = DataFlowTools.getDefSymbol(loop.getBody());
        return !def_symbol.contains(id.getSymbol());
    }

    /**
     * Check if the given expression is loop invariant
     */
    public static boolean isLoopInvariant(Loop loop, Expression e) {
        // Get def set for loop including loop header statement
        Set head_def_set =
                DataFlowTools.getDefSymbol(((ForLoop)loop).getStep());
        Set body_def_set = DataFlowTools.getDefSymbol(loop.getBody());
        Set<Symbol> accessed_set = SymbolTools.getAccessedSymbols(e);
        for (Symbol s : accessed_set) {
            if (head_def_set.contains(s) || body_def_set.contains(s)) {
                return false;
            }
        }
        return true;
    }

    /**
    * Returns symbols that may cause data dependences with the specified loop.
    * @param l the loop to be analyzed.
    * @return the set of involved symbols.
    */
    public static Set<Expression> collectScalarDependences(Loop l) {
        // SCALAR DEPENDENCE CHECK
        // -----------------------
        // Currently, test if for loops contain scalar variables in their def
        // set that are not marked private or reduction. These scalars may
        // cause dependences that we don't currently test for.
        // Temporary handling update for considering defined locations that 
        // are defined through the use of pointer arithmetic
        // This functionality must be handled by DataFlowTools getDefSymbols()
        // in the future
        Set<Expression> ret = new HashSet<Expression>();
        Set<Expression> def_exprs = DataFlowTools.getDefSet(l.getBody());
        Set<Expression> use_exprs = DataFlowTools.getUseSet(l.getBody());
        Map<Symbol, Expression> def_symbols = new HashMap<Symbol, Expression>();
        for (Expression def_expr : def_exprs) {
            if (def_expr instanceof AccessExpression ||
                def_expr instanceof Identifier ||
                def_expr instanceof ArrayAccess) {
                Symbol def_symbol = SymbolTools.getSymbolOf(def_expr);
                if (def_symbol != null) {
                    def_symbols.put(def_symbol, def_expr);
                } else {
                    ret.add(def_expr);
                }
            } else {
                ret.add(def_expr);
            }
        }
        for (Symbol sym : def_symbols.keySet()) {
            // If the variable is a scalar that is written to, and not marked
            // private or reduction
            if (SymbolTools.isScalar(sym) &&
                !isPrivate(sym, l) &&
                !isReduction(sym, l)) {
                // If the symbol is a pointer, check if all the definitions for
                // this pointer are array accesses, if not then scalar
                // dependence is possible. If yes, array data dependence
                // testing will handle it, i.e. return false
                if (SymbolTools.isPointer(sym)) {
                    for (Expression def : def_exprs) {
                        if (!(def instanceof ArrayAccess) &&
                            IRTools.containsSymbol(def, sym)) {
                            ret.add(def_symbols.get(sym));
                        }
                    }
                    // Check if the scalar is accessed via a struct symbol,
                    // even one member of which is an array or accessed as an
                    // array
                } else if (sym instanceof AccessSymbol) {
                    Symbol struct_sym = (AccessSymbol)sym;
                    while (struct_sym instanceof AccessSymbol) {
                        struct_sym = ((AccessSymbol)struct_sym).getBaseSymbol();
                        // The dependence analyzer will conservatively
                        // test for dependence across base symbols. Hence, 
                        // do nothing in this case, we don't consider this 
                        // to form a scalar dependence
                        if (SymbolTools.isArray(struct_sym)) {
                            break;
                            // If the base symbol is a pointer variable, check 
                            // if it is always accessed as an array. If yes, it 
                            // will be handled by the dependence analyzer
                        } else if (SymbolTools.isPointer(struct_sym)) {
                            for (Expression def : def_exprs) {
                                if (!(def instanceof ArrayAccess) &&
                                    struct_sym.equals(
                                            SymbolTools.getSymbolOf(def))) {
                                    ret.add(def_symbols.get(sym));
                                }
                            }
                            for (Expression use : use_exprs) {
                                if (!(use instanceof ArrayAccess) &&
                                    struct_sym.equals(
                                            SymbolTools.getSymbolOf(use))) {
                                    ret.add(def_symbols.get(sym));
                                }
                            }
                        } else {
                            ret.add(def_symbols.get(sym));
                        }
                    }
                } else {
                    ret.add(def_symbols.get(sym));
                }
            }
        }
        return ret;
    }

    /**
     * Check for scalars that are not privatizable or reduction variables
     */
    public static boolean scalarDependencePossible(Loop l) {
        // SCALAR DEPENDENCE CHECK
        // -----------------------
        // Currently, test if for loops contain scalar
        // variables in their def set that are not marked private or reduction.
        // These scalars may cause dependences that we don't currently test for.
        Set<Symbol> def_symbols = DataFlowTools.getDefSymbol(l.getBody());
        // Temporary handling update for considering defined locations that 
        // are defined through the use of pointer arithmetic
        // This functionality must be handled by DataFlowTools getDefSymbols()
        // in the future
        DFIterator<AssignmentExpression> dfs_iter =
                new DFIterator<AssignmentExpression>(
                        l.getBody(), AssignmentExpression.class);
        while (dfs_iter.hasNext()) {
            Expression lhs = dfs_iter.next().getLHS();
            if (lhs instanceof UnaryExpression &&
                        ((UnaryExpression)lhs).getOperator() ==
                        UnaryOperator.DEREFERENCE) {
                Expression unary = ((UnaryExpression)lhs).getExpression();
                def_symbols.addAll(SymbolTools.getAccessedSymbols(unary));
            }
        }
        for (Symbol sym : def_symbols) {
            // If the variable is a scalar that is written to, and not marked
            // private or reduction
            if ((SymbolTools.isScalar(sym)) &&
                (!(isPrivate(sym, l)) && !(isReduction(sym, l)))) {
                // If the symbol is a pointer, check if all the definitions for
                // this pointer are array accesses, if not then scalar
                // dependence is possible if yes, array data dependence testing
                // will handle it i.e. return false
                if (SymbolTools.isPointer(sym)) {
                    Set<Expression> def_set =
                            DataFlowTools.getDefSet(l.getBody());
                    for (Expression def : def_set) {
                        if (SymbolTools.getAccessedSymbols(def).contains(sym)) {
                            if (def instanceof ArrayAccess) {
                                continue;
                            } else {
                                return true;
                            }
                        }
                    }
                }
                // Check if the scalar is accessed via a struct symbol, even 
                // one member of which is an array or accessed as an array
                else if (sym instanceof AccessSymbol) {
                    Symbol struct_sym = (AccessSymbol)sym;
                    while (struct_sym instanceof AccessSymbol) {
                        struct_sym = ((AccessSymbol)struct_sym).getBaseSymbol();
                        if (SymbolTools.isArray(struct_sym)) {
                            // The dependence analyzer will conservatively
                            // test for dependence across base symbols. Hence, 
                            // do nothing in this case, we don't consider this 
                            // to form a scalar dependence
                            break;
                        } else if (SymbolTools.isPointer(struct_sym)) {
                            // If the base symbol is a pointer variable, check 
                            // if it is always accessed as an array. If yes, it 
                            // will be handled by the dependence analyzer
                            Set<Expression> def_set =
                                    DataFlowTools.getDefSet(l.getBody());
                            Set<Expression> use_set =
                                    DataFlowTools.getUseSet(l.getBody());
                            for (Expression def : def_set) {
                                if (struct_sym.equals(
                                        SymbolTools.getSymbolOf(def))) {
                                    if (def instanceof ArrayAccess) {
                                        continue;
                                    } else {
                                        return true;
                                    }
                                }
                            }
                            for (Expression use : use_set) {
                                if (struct_sym.equals(
                                        SymbolTools.getSymbolOf(use))) {
                                    if (use instanceof ArrayAccess) {
                                        continue;
                                    } else {
                                        return true;
                                    }
                                }
                            }
                        } else {
                            return true;
                        }
                    }
                }
                // if it isn't a pointer, scalar dependence does exist
                else {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Checks whether this loop contains any inner loops 
     */
    public static boolean isInnermostLoop(Loop loop) {
        if (containsLoop(loop)) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * Checks whether this loop is enclosed by any outer loops 
     */
    public static boolean isOutermostLoop(Loop loop) {
        if (loop instanceof ForLoop) {
            ForLoop for_loop = (ForLoop)loop;
            Traversable t = for_loop.getParent();
            while (t != null) {
                if (t instanceof ForLoop) {
                    return false;
                } else {
                    t = t.getParent();
                }
            }
        }
        return true;
    }

    /**
     * Get the outermost loop for the nest that surrounds the input loop
     */
    public static Loop getOutermostLoop(Loop loop) {
        Loop return_loop = null;
        if (loop instanceof ForLoop) {
            if (isOutermostLoop(loop)) {
                return_loop = loop;
            } else {
                ForLoop for_loop = (ForLoop)loop;
                Traversable t = for_loop.getParent();
                while (t != null) {
                    if (t instanceof ForLoop) {
                        if (isOutermostLoop((Loop)t)) {
                            return_loop = (Loop)t;
                        } else {
                            t = t.getParent();
                        }
                    } else {
                        t = t.getParent();
                    }
                }
            }
        }
        return return_loop;
    }

    /**
     * Check whether the loop contains control constructs that cause it to
     * terminate before the loop condition is reached. Check only at one
     * nesting level, no need to check for control flow constructs in nested
     * loops
     */
    public static boolean containsControlFlowModifier(Loop loop) {
        if (loop instanceof ForLoop) {
            DFIterator<Statement> iter =
                    new DFIterator<Statement>(loop.getBody(), Statement.class);
            iter.pruneOn(Loop.class);
            while (iter.hasNext()) {
                Statement stmt = iter.next();
                if (stmt instanceof GotoStatement ||
                    stmt instanceof BreakStatement ||
                    stmt instanceof Label ||
                    stmt instanceof ReturnStatement) {
                    return true;
                }
            }
            return false;
        } else {
            return true;
        }
    }

    /**
     * Check if the loop contains only a break statement modifier
     */
    public static boolean containsBreakStatement(Loop loop) {
        if (loop instanceof ForLoop) {
            DFIterator<BreakStatement> iter = new DFIterator<BreakStatement>(
                    loop.getBody(), BreakStatement.class);
            iter.pruneOn(Loop.class);
            iter.reset(); // prune is not applied during construction.
            return iter.hasNext();
        } else {
            return false;
        }
    }

    public static boolean
            containsControlFlowModifierOtherThanBreakStmt(Loop loop) {
        if (loop instanceof ForLoop) {
            DFIterator<Statement> iter =
                    new DFIterator<Statement>(loop.getBody(), Statement.class);
            iter.pruneOn(Loop.class);
            while (iter.hasNext()) {
                Statement stmt = iter.next();
                if (stmt instanceof GotoStatement ||
                    stmt instanceof Label ||
                    stmt instanceof ReturnStatement) {
                    return true;
                }
            }
            return false;
        } else {
            return true;
        }
    }

  /**
   * Calculate the nest of loops enclosed within the given loop
   */
    public static LinkedList<Loop >
        calculateInnerLoopNest(Loop enclosing_loop) {
        LinkedList<Loop> ret_nest = new LinkedList<Loop>();
        DepthFirstIterator<Traversable> dfs_iter =
            new DepthFirstIterator<Traversable >
            ((Traversable)enclosing_loop);
        for (;;) {
            Loop l = null;
            try {
                l = (Loop)dfs_iter.next(Loop.class);
                ret_nest.add(l);
            }
            catch(NoSuchElementException e) {
                break;
            }
        }
        return ret_nest;
    }

    /**
     * Returns the loop name inserted by Cetus.
     * @param loop the loop.
     * @return the loop name if one exists, null otherwise.
     */
    public static String getLoopName(Statement loop) {
        PragmaAnnotation note =
                loop.getAnnotation(PragmaAnnotation.class, "name");
        return (note == null) ? null : (String)note.get("name");
    }

    /**
     * Adds a unique loop name to each for loop.
     * @param program the input program.
     */
    public static void addLoopName(Program program) {
        if (!is_loop_named) {
            addLoopName(program, "", null);
        }
        is_loop_named = true;
    }

    /* Adds a unique loop name to each for loop - recursive call */
    private static void addLoopName(
            Traversable tr, String header, LinkedList<Integer> nums) {
        Map<Statement, PragmaAnnotation> names =
                new HashMap<Statement, PragmaAnnotation>();
        List<Traversable> children = tr.getChildren();
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable t = children.get(i);
            if (t == null) {
                ;
            } else if (t instanceof Procedure) {
                LinkedList<Integer> init_nums =
                        new LinkedList<Integer>(Arrays.asList(new Integer(0)));
                addLoopName(t, header + ((Procedure)t).getName(), init_nums);
            } else if (t instanceof ForLoop) {
                Statement loop = (Statement)t;
                String my_name =
                        header + "#" + PrintTools.listToString(nums, "#");
                PragmaAnnotation note = new PragmaAnnotation("loop");
                note.put("name", my_name);
                loop.annotate(note);
                nums.add(new Integer(0));
                addLoopName(t, header, nums);
                nums.removeLast();
                nums.set(nums.size() - 1, nums.getLast() + 1);
            } else if (t.getChildren() != null) {
                addLoopName(t, header, nums);
            }
        }
    }

    /**
     * Check if the input symbol is marked as private to the loop by the
     * Privatization pass. ArrayPrivatization MUST be run before using this
     * test.
     * @param s the symbol which needs to be checked as private to loop or not
     * @param l the loop with respect to which private property needs to be
     *      checked
     * @return true if private, else false
     */
    public static boolean isPrivate(Symbol s, Loop l) {
        Annotatable an = (Annotatable)l;
        CetusAnnotation note =
                an.getAnnotation(CetusAnnotation.class, "private");
        if (note != null) {
            Set<Symbol> private_symbols = note.get("private");
            if (private_symbols.contains(s)) {
                return true;
            }
        }
        note = an.getAnnotation(CetusAnnotation.class, "lastprivate");
        if (note != null) {
            Set<Symbol> private_symbols = note.get("lastprivate");
            if (private_symbols.contains(s)) {
                return true;
            }
        }
        note = an.getAnnotation(CetusAnnotation.class, "firstprivate");
        if (note != null) {
            Set<Symbol> private_symbols = note.get("firstprivate");
            if (private_symbols.contains(s)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Check if the input symbol is marked as reduction for the loop by the
    * Reduction pass. Reduction Analysis MUST be run before using this test
    * @param s the symbol which needs to be checked as reduction variable or not
    * @param l the loop with respect to which reduction property needs to be
    *       checked
    * @return true if reduction variable, else false
    */
    public static boolean isReduction(Symbol s, Loop l) {
        CetusAnnotation note = ((Annotatable)l).getAnnotation(
                CetusAnnotation.class, "reduction");
        if (note != null) {
            Map<String, Set<Expression>> m = note.get("reduction");
            for (String op : m.keySet()) {
                Set<Expression> ts = m.get(op);
                for (Expression re : ts) {
                    if (s.equals(SymbolTools.getSymbolOf(re))) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
    * Checks the eligibility of a certain loop for data dependence testing
    * Eligibility checks can be added or removed to increase the scope of
    * dependence testing
    * @param loop
    * @return true if the loop is formed well enough to be eligible.
    */
    public static boolean checkDataDependenceEligibility(Loop loop) {
        // Checks whether the loop is in a conventional
        // Fortran type do loop format. Symbolic values are allowed in loop
        // header statement
        if (isCanonical(loop) == false) {
            PrintTools.println("Loop is not canonical", 4);
            return false;
        }
        // One of the checks to see that the loop body contains no side-effects
        if (containsFunctionCall(loop) == true) {
            if (containsOnlyParallelizableCall(loop) == false) {
                PrintTools.println("Loop can have side-effects", 4);
                return false;
            }
        }
        // Deal only with perfectly nested loops
        /*
        if (isPerfectNest(loop)==false) {
            PrintTools.println("Loop does not contain a perfect nest", 4);
            return false;
        }
        */
        if (containsControlFlowModifierOtherThanBreakStmt(loop) == true) {
            PrintTools.println("Loop contains control flow modifiers", 4);
            return false;
        }
        if (isIncrementEligible(loop) == false) {
            PrintTools.println(
                    "Loop contains indeterminate symbolic increment", 4);
            return false;
        }
        return true;
    }

    /**
    * Returns true if the loop increment is an integer constant value. If
    * symbolic, uses range information to determine if it is an integer
    * constant value or not
    * @param loop the loop whose increment needs to be checked
    * @return true if increment is an integer constant value
    */
    public static boolean isIncrementEligible(Loop loop) {
        boolean eligible_inc = true;
        if (isIncrementConstant(loop) == false) {
            RangeDomain loop_range =
                    RangeAnalysis.getRangeDomain((ForLoop)loop);
            Expression curr_inc = getIncrementExpression(loop);
            Set<Symbol> loop_stmt_symbols = loop_range.getSymbols();
            Expression new_inc =
                    loop_range.substituteForward(curr_inc, loop_stmt_symbols);
            if (!(new_inc instanceof IntegerLiteral)) {
                eligible_inc = false;
            } else {
                eligible_inc = true;
            }
        }
        return eligible_inc;
    }

    /**
    * Get symbol of loop index, index variable is identified from step
    * expression for loop
    * @return Symbol for the loop index variable
    */
    public static Symbol getLoopIndexSymbol(Loop loop) {
        Expression indexVar = getIndexVariable(loop);
        return SymbolTools.getSymbolOf(indexVar);
    }

    /**
    * Replaces symbolic values in loop lower bound with information obtained
    * from range analysis. If symbolic bound has a constant value, that value
    * is the new lower bound. In case of a range expression, the lowerbound of
    * the range is conservatively used as the lowerbound for the loop. If the
    * symbolic value has an indeterminate value, the minimum possible value for
    * Long is assigned to the lower bound
    * @param loop the loop whose symbolic lower bound is to be replaced
    * @param loop_rd the range domain representing ranges for all vars in the
    *       loop statement
    * @return the expression for the new lowerbound with constant value
    */
    public static Expression
            replaceSymbolicLowerBound(Loop loop, RangeDomain loop_rd) {
        Expression new_lb = null;
        long EXTREME_VALUE = Integer.MIN_VALUE;
        Identifier index = (Identifier)getIndexVariable(loop);
        if (checkIncrementExpression(loop, index) == 1) {
            EXTREME_VALUE = Integer.MIN_VALUE;
        } else if (checkIncrementExpression(loop, index) == -1) {
            EXTREME_VALUE = Integer.MAX_VALUE;
        }
        Set<Symbol> loop_stmt_symbols = loop_rd.getSymbols();
        Expression curr_lb = getLowerBoundExpression(loop);
        new_lb = loop_rd.expandSymbols(curr_lb, loop_stmt_symbols);
        if (new_lb instanceof RangeExpression) {
            Expression lb_of_range = ((RangeExpression)new_lb).getLB();
            if (lb_of_range instanceof IntegerLiteral) {
                new_lb = lb_of_range;
            } else {
                new_lb = new IntegerLiteral(EXTREME_VALUE);
            }
        } else if (new_lb instanceof Expression) {
            new_lb = new IntegerLiteral(EXTREME_VALUE);
        }
        return new_lb;
    }

    /**
    * Replaces symbolic values in loop lower bound with information obtained
    * from range analysis. If symbolic bound has a constant value, that value
    * is the new lower bound. In case of a range expression, the lowerbound of
    * the range is conservatively used as the lowerbound for the loop. If the
    * symbolic value has an indeterminate value, the minimum possible value for
    * Long is assigned to the lower bound
    * @param loop the loop whose symbolic lower bound is to be replaced
    * @return the expression for the new lowerbound with constant value
    */
    public static Expression replaceSymbolicLowerBound(Loop loop) {
        RangeDomain loop_range = RangeAnalysis.getRangeDomain((Statement)loop);
        return (replaceSymbolicLowerBound(loop, loop_range));
    }

    /**
    * Replaces symbolic values in loop upper bound with information obtained
    * from range analysis. If symbolic bound has a constant value, that value
    * is the new upper bound. In case of a range expression, the upperbound of
    * the range is conservatively used as the upperbound for the loop. If the
    * symbolic value has an indeterminate value, the maximum possible value for
    * Long is assigned to the upper bound
    * @param loop the loop whose symbolic upper bound is to be replaced
    * @param loop_rd the range domain representing ranges for all vars in the
    *       loop statement
    * @return the expression for the new upperbound with constant value
    */
    public static Expression
            replaceSymbolicUpperBound(Loop loop, RangeDomain loop_rd) {
        Expression new_ub = null;
        long EXTREME_VALUE = Integer.MAX_VALUE;
        Identifier index = (Identifier)getIndexVariable(loop);
        if (checkIncrementExpression(loop, index) == 1) {
            EXTREME_VALUE = Integer.MAX_VALUE;
        } else if (checkIncrementExpression(loop, index) == -1) {
            EXTREME_VALUE = Integer.MIN_VALUE;
        }
        Set<Symbol> loop_stmt_symbols = loop_rd.getSymbols();
        Expression curr_ub = getUpperBoundExpression(loop);
        new_ub = loop_rd.expandSymbols(curr_ub, loop_stmt_symbols);
        if (new_ub instanceof RangeExpression) {
            Expression ub_of_range = ((RangeExpression)new_ub).getUB();
            if (ub_of_range instanceof IntegerLiteral) {
                new_ub = ub_of_range;
            } else {
                new_ub = new IntegerLiteral(EXTREME_VALUE);
            }
        } else if (new_ub instanceof Expression) {
            new_ub = new IntegerLiteral(EXTREME_VALUE);
        }
        return new_ub;
    }

    /**
    * Replaces symbolic values in loop upper bound with information obtained
    * from range analysis. If symbolic bound has a constant value, that value
    * is the new upper bound. In case of a range expression, the upperbound of
    * the range is conservatively used as the upperbound for the loop. If the
    * symbolic value has an indeterminate value, the maximum possible value for
    * Long is assigned to the upper bound
    * @param loop the loop whose symbolic upper bound is to be replaced
    * @return the expression for the new upperbound with constant value
    */
    public static Expression replaceSymbolicUpperBound(Loop loop) {
        RangeDomain loop_range = RangeAnalysis.getRangeDomain((Statement)loop);
        return (replaceSymbolicUpperBound(loop, loop_range));
    }

    public static List<Loop>
            extractOutermostDependenceTestEligibleLoops(Traversable t) {
        List<Loop> ret = new ArrayList<Loop>(4);
        // Iterate depth-first over all loops including the enclosing loop
        // Identify all nests eligible for testing, including arbitrarily
        // nested loops
        List<Loop> tested_loops = new ArrayList<Loop>(4);
        DFIterator<Loop> iter = new DFIterator<Loop>(t, Loop.class);
        while (iter.hasNext()) {
            Loop loop = iter.next();
            boolean nest_eligible = false;
            if (!tested_loops.contains(loop)) {
                LinkedList<Loop> nest = LoopTools.calculateInnerLoopNest(loop);
                for (Loop l : nest) {
                    nest_eligible = LoopTools.checkDataDependenceEligibility(l);
                    if (!nest_eligible) {
                        break;
                    }
                }
                if (nest_eligible) {
                    tested_loops.addAll(nest);
                    ret.add(loop);
                }
            }
        }
        return ret;
    }

    /**
    * A simple implementation based on the Euclidean GCD algorithm to 
    * calculate the distance in terms of loop iterations between 
    * subscripts of dependent arrays. Only handles single index variable 
    * subscripts that are affine expressions of that single loop index 
    * variable. Does not handle coupled subscripts
    * @param loop the loop w.r.t which distance needs to be calculated
    * @param e1 array access that is one end of the dependence
    * @param e2 array access that is at the other end of the dependence
    */
    public static long
            getReuseDistance(Loop loop, ArrayAccess e1, ArrayAccess e2) {
        // Integer.MAX_VALUE is the value used if distance is UNKNOWN
        long reuse_distance = Integer.MAX_VALUE;
        // Loop information for calculating the distance
        long lowerbound, upperbound, increment;
        // Iteration points for the first and second iterations
        long x, y;
        // Number of distance values measured
        long distance;
        // Value holders
        long na, nb, gcd;

        // Eligibility checks on the loop
        if (isCanonical(loop) == false || isIncrementEligible(loop) == false) {
            return reuse_distance;
        }

        // If the two loops don't have equal number of indices, return
        if (e1.getNumIndices() != e2.getNumIndices()) {
            return reuse_distance;
        }

        // If the dependent array accesses are aliased, then we give up here 
        // and return the dependence distance as unknown
        Symbol s1 = SymbolTools.getSymbolOf(e1);
        Symbol s2 = SymbolTools.getSymbolOf(e2);
        if (!(s1.equals(s2))) {
            return reuse_distance;
        }

        // Obtain a range domain for the loop being considered
        RangeDomain loop_range = RangeAnalysis.getRangeDomain((Statement)loop);
        LoopInfo loop_info = new LoopInfo(loop);
        // Lower bound for loop is not constant, use range information
        if (!(LoopTools.isLowerBoundConstant(loop))) {
            Expression new_lb =
                    LoopTools.replaceSymbolicLowerBound(loop, loop_range);
            // Assign new lower bound
            lowerbound = ((IntegerLiteral)new_lb).getValue();
        } else {
            lowerbound = ((IntegerLiteral)(loop_info.getLoopLB())).getValue();
        }
        // Upper bound for loop is not constant, use range information
        if (!(LoopTools.isUpperBoundConstant(loop))) {
            Expression new_ub =
                    LoopTools.replaceSymbolicUpperBound(loop, loop_range);
            // Assign new upper bound
            upperbound = ((IntegerLiteral)new_ub).getValue();
        } else {
            upperbound = ((IntegerLiteral)(loop_info.getLoopUB())).getValue();
        }
        // Increment for loop is not constant, use range information
        // Range information will return constant integer increment value as
        // the loop has already been considered eligible for dependence testing
        if (!(LoopTools.isIncrementConstant(loop))) {
            Expression curr_inc = loop_info.getLoopIncrement();
            Set<Symbol> loop_stmt_symbols = loop_range.getSymbols();
            Expression new_inc =
                    loop_range.substituteForward(curr_inc, loop_stmt_symbols);
            loop_info.setLoopIncrement(new_inc);
            increment = ((IntegerLiteral)new_inc).getValue();
        } else {
            increment =
                    ((IntegerLiteral)(loop_info.getLoopIncrement())).getValue();
        }
        // For negative step, exchange loop bounds and convert to positive step
        if (increment < 0) {
            increment *= -1;
            long temp = lowerbound;
            lowerbound = upperbound;
            upperbound = temp;
        }

        Identifier loop_id = (Identifier)loop_info.getLoopIndex();

        ArrayList<Identifier> ids = new ArrayList<Identifier>();
        ids.add(loop_id);

        int dimensions = e1.getNumIndices();
        // Go through each subscript, currently we're handling only SIV
        // and non-coupled subscripts. So not more than one of the subscripts
        // should be associated with the required loop variable
        for (int i = 0; i < dimensions; i++) {
            Expression e1_subscript = e1.getIndex(i);
            Expression e2_subscript = e2.getIndex(i);
            long triplet[];

            if (Symbolic.isAffine(e1_subscript, ids) &&
                Symbolic.isAffine(e2_subscript, ids)) {
                Expression a = Symbolic.getCoefficient(e1_subscript, loop_id);
                Expression b = Symbolic.getCoefficient(e2_subscript, loop_id);
                Expression c1 =
                        Symbolic.getConstantCoefficient(e1_subscript, ids);
                Expression c2 =
                        Symbolic.getConstantCoefficient(e2_subscript, ids);
                Expression diff = Symbolic.subtract(c2, c1);

                if (a instanceof IntegerLiteral &&
                    b instanceof IntegerLiteral &&
                    c1 instanceof IntegerLiteral &&
                    c2 instanceof IntegerLiteral) {
                    long a_value = ((IntegerLiteral)a).getValue();
                    long b_value = ((IntegerLiteral)b).getValue();
                    long c1_value = ((IntegerLiteral)c1).getValue();
                    long c2_value = ((IntegerLiteral)c2).getValue();
                    long diff_value = ((IntegerLiteral)diff).getValue();
                    // If either of the coefficients of the loop index is 0, 
                    // the dependence distance is variable, hence return UNKNOWN
                    if (a_value == 0 || b_value == 0) {
                        break;
                    }
                    // Get the GCD triplet for the coefficients using the
                    // Extended Euclidean algorithm
                    triplet =
                        GCD.computeWithLinearCombination(a_value, b_value);
                    gcd = triplet[0];
                    na = triplet[1];
                    nb = triplet[2];
                    // If the constant in the expression is completely
                    // divisible by the GCD, a dependence exists and we can try
                    // to calculate the distance
                    if (gcd != 0 && diff_value % gcd == 0) {
                        distance = Integer.MAX_VALUE;
                        for (int k = 1; k < 10; k++) {
                            x = na*(diff_value/gcd) + k*(b_value/gcd);
                            y = nb*(diff_value/gcd) + k*(a_value/gcd);
                            if ((x >= lowerbound && x <= upperbound)
                                && (y >= lowerbound && y <= upperbound)) {
                                // If divisible by increment
                                if ((y - x)%increment == 0) {
                                    distance = (y - x)/increment;
                                }
                                break;
                            }
                        }
                        if (distance == Integer.MAX_VALUE) {
                            for (int k = -1; k > -10; k--) {
                                x = na*(diff_value/gcd) + k*(b_value/gcd);
                                y = nb*(diff_value/gcd) + k*(a_value/gcd);
                                if ((x >= lowerbound && x <= upperbound) &&
                                    (y >= lowerbound && y <= upperbound)) {
                                    // If divisible by increment
                                    if ((y - x)%increment == 0) {
                                        distance = (y - x) / increment;
                                    }
                                    break;
                                }
                            }
                        }
                        reuse_distance = distance;
                    }
                }
            }
            // This subscript pair is not affine with respect to the required
            // loop variable, so don't assign a reuse distance, move to the
            // next subscript pair
        }
        return reuse_distance;
    }

    /**
    * Returns the control structure of the specified loop in string.
    * 
    * @param loop the given loop.
    * @return the string for the control structure.
    */
    public static String toControlString(Loop loop) {
        StringBuilder str = new StringBuilder(80);
        if (loop instanceof ForLoop) {
            ForLoop forloop = (ForLoop)loop;
            Statement init = forloop.getInitialStatement();
            Expression condition = forloop.getCondition();
            Expression step = forloop.getStep();
            str.append("for (");
            if (init != null) {
                str.append(init);
            }
            if (condition != null) {
                str.append(" ").append(condition);
            }
            str.append("; ");
            if (step != null) {
                str.append(step);
            }
            str.append(")");
        } else if (loop instanceof WhileLoop) {
            str.append("while (").append(loop.getCondition()).append(")");
        } else if (loop instanceof DoLoop) {
            str.append("do..while (").append(loop.getCondition()).append(")");
        }
        return str.toString();
    }
}
