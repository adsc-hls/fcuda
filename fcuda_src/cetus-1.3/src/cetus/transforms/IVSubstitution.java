package cetus.transforms;

import cetus.analysis.*;
import cetus.hir.*;
import java.util.*;

/**
 * IVSubstitution performs induction variable substitution for additive
 * induction variables. The transformation proceeds in two steps, analysis and
 * transformation. The analysis phase detects potential induction variables
 * by matching the statement type that have the following structure.
 *   iv = iv+expr, iv = iv-expr
 *   iv += expr, iv -= expr
 *   iv++, iv--, ++iv, --iv
 * , where expr is either loop-invariant or another induction variable.
 * iv, of course, cannot appear as a defined variable in the loop except for
 * the induction statement. After this matching process, comes a feasibility
 * test that removes any cases where substitution cannot proceed.
 *
 * The transformation phase performs substitution and statement removal
 * depending on the type of each statement. The key enabling functionality of
 * this phase is symbolic summation operation which is provided as a external
 * tool implemented in Symbolic class.
 *
 * TODOs:
 * incorporation with live variable analysis.
 *
 * Algorithms in the CPC09 paper.
 */
public class IVSubstitution extends TransformPass {
    /**
    * Set of complex statements that should be avoided.
    */
    private static final Set<Class<? extends Traversable>> fail_fast_class;

    /**
    * Flags for cases handled or not handled by Cetus IV pass.
    */
    private static final int PASS = 0;  // Everything is o.k.
    private static final int FAIL = -1; // Transformation should stop.
    private static final int WRAP = 1;  // Wrap-around variable detected.

    /**
    * Debug level.
    */
    private static final int debug = PrintTools.getVerbosity();

    /**
    * Debug tag.
    */
    private static final String tag = "[IVSubstitution]";

    static {
        fail_fast_class = new HashSet<Class<? extends Traversable>>();
        fail_fast_class.add(SwitchStatement.class);
        fail_fast_class.add(DoLoop.class);
        fail_fast_class.add(WhileLoop.class);
        fail_fast_class.add(BreakStatement.class);
        fail_fast_class.add(ContinueStatement.class);
        fail_fast_class.add(Label.class);
        fail_fast_class.add(GotoStatement.class);
        fail_fast_class.add(ReturnStatement.class);
        fail_fast_class.add(FunctionCall.class);
    }

    /**
    * Loops eligible for transformation. Loops are eligible for transformation
    * if it is always visited since the entry of the outer loop.
    */
    private LinkedList<Loop> eligible_loops = new LinkedList<Loop>();

    /**
    * Map from loops to their lb, ub, and step after forward substitution.
    */
    private Map<Loop, List<Expression>> loop_to_bounds =
            new LinkedHashMap<Loop, List<Expression>>();

    /**
    * Map from loops to their child induction statements; data of this map
    * is another map from each induction statmenet to its detected increment
    * expression.
    */
    private Map<Loop, Map<Statement, Expression>> loop_to_stmts =
            new LinkedHashMap<Loop, Map<Statement, Expression>>();

    /**
    * Place for storing the increment expression before forward substitution.
    * Original form of the increments are used to recognize non-induction.
    */
    private Map<Statement, Expression> ivstmt_to_inc =
            new LinkedHashMap<Statement, Expression>();

    /**
    * Map from loops to their variants; any variable definitions other than
    * induction statement, loop index initialization are regarded as variants.
    */
    private Map<Loop, Set<Symbol>> loop_to_variants =
            new LinkedHashMap<Loop, Set<Symbol>>();

    /**
    * Map from loops to their IV candidates. These candidates mean that the
    * substitution must start from the associated loops implying that a loop
    * can have an induction statement but no candidates.
    */
    private Map<Loop, Set<Symbol>> loop_to_ivs =
            new LinkedHashMap<Loop, Set<Symbol>>();

    private Map<Loop, Set<Statement>> loop_to_wrap_stmts =
            new LinkedHashMap<Loop, Set<Statement>>();

    /**
    * Map from induction variables to their divisibility properties. This is
    * useful when computing closed form expressions since more aggressive
    * expression manipulation is possible with divisibility property.
    */
    private Map<Symbol, Boolean> iv_to_divisible =
            new LinkedHashMap<Symbol, Boolean>();

    /**
    * Map form induction variables to the set of loops that enclose the
    * variables' induction statements.
    */
    private Map<Symbol, Set<Loop>> iv_to_loops =
            new LinkedHashMap<Symbol, Set<Loop>>();

    /**
    * Map from loop and induction statements to their increments after. These
    * increments are cumulative since the entry to the loop.
    */
    private Map<Statement, Expression> inc_after_stmt =
            new LinkedHashMap<Statement, Expression>();

    /**
    * Map from loops to their increments into. These increments are used to
    * replace induction variables that appear before the induction statements.
    */
    private Map<Loop, Expression> inc_into_loop =
            new LinkedHashMap<Loop, Expression>();

    /**
    * Constructs an induction variable transformation pass with the given
    * program object.
    * @param program the input program.
    */
    public IVSubstitution(Program program) {
        super(program);
    }

    /**
    * Returns the pass name for the transformation.
    * @return the pass name.
    */
    public String getPassName() {
        return tag;
    }

    /**
    * Performs the induction variable substitution transformation in two phases.
    * First phase performs analysis to check if the loop is eligible for
    * transformation, and the second phase performs actual transformation if
    * it is allowed.
    */
    public void start() {
        DFIterator<Traversable> iter = new DFIterator<Traversable>(program);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(DeclarationStatement.class);
        iter.pruneOn(ExpressionStatement.class);
        iter.pruneOn(ForLoop.class);
        while (iter.hasNext()) {
            Traversable t = iter.next();
            if (t instanceof TranslationUnit) {
                PrintTools.printlnStatus(1, tag, "Entering translation unit",
                        "\"", ((TranslationUnit)t).getInputFilename(), "\"");
            } else if (t instanceof Procedure) {
                Procedure proc = (Procedure)t;
                PrintTools.printlnStatus(1, tag, "Entering procedure",
                        "\"", proc.getName(), "\"");
            } else if (t instanceof ForLoop) {
                ForLoop loop = (ForLoop)t;
                int result = analyze(loop);
                if (result == WRAP) {
                    // Wrap-around variable has been detected.
                    removeWrapAroundVariables();
                    result = analyze(loop);
                }
                if (result == PASS) {
                    // Transform the loop if analysis is successful
                    transform(loop);
                }
            }
        }
    }

    /**
    * Performs induction variable analysis.
    * Analysis performs the following tasks.
    * 1. Checks if the loops are structured well. <br>
    * 2. Collects candidate induction statements. <br>
    * 3. Schedule transformation by placing induction variables at the right
    * location in the loop nest; they are moved outer level as long as there
    * are no destructive modifications to the induction variables. <br>
    * @param loop the outer-most loop for the analysis.
    * @return PASS if iv substitution can progress, FAIL otherwise.
    */
    private int analyze(Loop loop) {
        if (IRTools.containsClasses(loop, fail_fast_class)) {
            PrintTools.printlnStatus(1, tag, 
                    "Analysis stops due to an intractable loop:");
            PrintTools.printlnStatus(1, getReport(loop));
            return FAIL;
        }

        // Clean up analysis results from the previous outer-loop analysis.
        eligible_loops.clear();
        loop_to_bounds.clear();
        loop_to_stmts.clear();
        ivstmt_to_inc.clear();
        loop_to_variants.clear();
        loop_to_ivs.clear();
        loop_to_wrap_stmts.clear();
        iv_to_divisible.clear();
        iv_to_loops.clear();

        if (matchInduction(loop) != PASS) {
            return FAIL;
        }

        // Initialization & bound extraction.
        for (Loop curr : eligible_loops) {
            loop_to_ivs.put(curr, new LinkedHashSet<Symbol>());
            // Stores forward-substituted loop steps and bounds.
            RangeDomain rd = RangeAnalysis.query((Statement)curr);
            List<Expression> bounds = new LinkedList<Expression>();
            bounds.add(LoopTools.getIndexVariable(curr));
            bounds.add(rd.substituteForward(
                    LoopTools.getLowerBoundExpression(curr)));
            bounds.add(rd.substituteForward(
                    LoopTools.getUpperBoundExpression(curr)));
            bounds.add(rd.substituteForward(
                    LoopTools.getIncrementExpression(curr)));
            loop_to_bounds.put(curr, bounds);
        }
    
        // Places induction variables at the outer-most loop that does not
        // contain any destructve updates (modifications other than induction)
        // to the induction variables. For exmaple,
        // 1. for (i=...)
        //      j = ...
        // 2.   for (k=...)
        //        j += ...
        //        l += ...
        // The owner of j is loop2 while that of l is loop1.
        // iv_placed contains induction variables placed at inner loops due
        // to increments and relevant loop bounds found to be loop-variant.
        // If one of the loops where a certain iv is placed is an inner loop of
        // the current loop, the induction statement of the current loop is
        // treated as a side effect and no placement occurs.
        Map<Symbol, Statement> iv_placed =
                new LinkedHashMap<Symbol, Statement>();
        for (int i = 0; i < eligible_loops.size(); i++) {
            Loop curr = eligible_loops.get(i);
            if (i > 0 && !IRTools.isAncestorOf(curr, eligible_loops.get(i-1))) {
                iv_placed.clear();
            }

            Map<Statement, Expression> iv_stmts = loop_to_stmts.get(curr);
            Set<Symbol> variants = loop_to_variants.get(curr);
            // Iteratively remove any unsafe induction statements from the
            // candidate list. If there is any destructive updates to the
            // induction variables the size of variants keep increasing.
            int variants_size = -1;
            while (variants.size() != variants_size) {
                variants_size = variants.size();
                for (Statement stmt :
                        new LinkedHashSet<Statement>(iv_stmts.keySet())) {
                    Symbol iv = getIV(stmt);
                    if (iv_placed.get(iv) != null &&
                        IRTools.isAncestorOf(curr, iv_placed.get(iv)) ||
                        variants.contains(iv) ||
                        IRTools.containsSymbols(ivstmt_to_inc.get(stmt),
                                                variants)) {
                        iv_stmts.remove(stmt);
                        variants.add(iv);
                    }
                }
            }

            // Place the induction variables at the right location.
            // Starting from each induction statement, it traverses upwards to
            // detect any destructive updates to variables that affect the
            // computation of the closed-form expressions of each induction
            // variables. When there is such program semantics, the current
            // induction variable should be scheduled for closed-form expression
            // computation at the inner loops of the loop that contains such
            // destructive semantics.
            for (Statement stmt : iv_stmts.keySet()) {
                Symbol iv = getIV(stmt);
                Set<Symbol> vars_with_ivs = new LinkedHashSet<Symbol>();
                // Add iv and variables in the increment expression in
                // vars_with_ivs.
                vars_with_ivs.add(iv);
                vars_with_ivs.addAll(
                        SymbolTools.getAccessedSymbols(iv_stmts.get(stmt)));
                Traversable t = stmt;
                Loop outer = null;
                while (true) {
                    if (t instanceof Loop && eligible_loops.contains(t)) {
                        Loop curr_loop = (Loop)t;
                        List<Expression> loop_bounds =
                                loop_to_bounds.get(curr_loop);
                        Set<Symbol> loop_variants = new LinkedHashSet<Symbol>(
                                loop_to_variants.get(curr_loop));
                        // Exclude current index from loop-variant inner bounds
                        // since the loop structure is canonical.
                        vars_with_ivs.remove(
                                ((Identifier)loop_bounds.get(0)).getSymbol());
                        // Exclude wrap-around variables only if curr_loop is
                        // an nzt loop.
                        if (getNZTCondition(curr_loop) == null) {
                            Iterator<Symbol> iter = vars_with_ivs.iterator();
                            while (iter.hasNext()) {
                                if (isWrapAroundVariable(
                                        curr_loop, iter.next())) {
                                    iter.remove();
                                }
                            }
                        }
                        // Should schedule the transformation at the current
                        // loop.
                        if (IRTools.containsSymbols(
                                    loop_variants, vars_with_ivs)) {
                            break;
                        }
                        outer = curr_loop;
                        // Include variables in loop bounds and steps in
                        // vars_with_ivs.
                        vars_with_ivs.addAll(SymbolTools.getAccessedSymbols(
                                loop_bounds.get(1)));
                        vars_with_ivs.addAll(SymbolTools.getAccessedSymbols(
                                loop_bounds.get(2)));
                        vars_with_ivs.addAll(SymbolTools.getAccessedSymbols(
                                loop_bounds.get(3)));
                        if (outer == eligible_loops.getLast()) {
                            break;
                        }
                    }
                    t = t.getParent();
                }
                // Record iv placements
                loop_to_ivs.get(outer).add(iv);
                iv_placed.put(iv, (Statement)outer);
                // Update divisibilities
                Boolean is_divisible = iv_to_divisible.get(iv);
                if (is_divisible == null || is_divisible == true) {
                    if (IRTools.containsBinary(stmt, BinaryOperator.DIVIDE) ||
                        IRTools.containsBinary(stmt, BinaryOperator.MODULUS)) {
                        iv_to_divisible.put(iv, new Boolean(false));
                    } else {
                        iv_to_divisible.put(iv, new Boolean(true));
                    }
                }
                // Fill iv_to_loop map.
                Set<Loop> iv_loops = iv_to_loops.get(iv);
                if (iv_loops == null) {
                    iv_loops = new LinkedHashSet<Loop>();
                    iv_to_loops.put(iv, iv_loops);
                }
                iv_loops.add(curr);
            }
        }
        // Print the summary of the analysis.
        if (debug >= 3) {
            PrintTools.printlnStatus(3, tag, "Analysis result:");
            reportAnalysis(loop, "");
        }
        return (loop_to_wrap_stmts.isEmpty())? PASS: WRAP;
    }

    /**
    * Returns a condition under which the specified loop is a non-zero trip
    * loop.
    * @param loop the given loop.
    * @return null if it is proved to be non-zero-trip loop statically, the
    * computed condition otherwise.
    */
    private Expression getNZTCondition(Loop loop) {
        Expression ret = null;
        RangeDomain rd = RangeAnalysis.query((Statement)loop);
        Expression lb = LoopTools.getLowerBoundExpression(loop);
        Expression ub = LoopTools.getUpperBoundExpression(loop);
        if (!rd.compare(lb, ub).isLE()) {
            ret = Symbolic.le(lb, ub);
        }
        return ret;
    }

    /**
    * Returns a condition under which all loops below the specified loop
    * executes at least once.
    * @param loop the given loop.
    * @return null if it is statically proved, the computed condition otherwise.
    */
    private Expression getNZTConditions(Loop loop) {
        Expression ret = null;
        for (int i = 0; i <= eligible_loops.indexOf(loop); i++) {
            Loop curr_loop = eligible_loops.get(i);
            if (IRTools.isAncestorOf(loop, curr_loop) &&
                !loop_to_ivs.get(curr_loop).isEmpty()) {
                Expression curr_condition = getNZTCondition(curr_loop);
                if (curr_condition != null) {
                    if (ret == null) {
                        ret = curr_condition;
                    } else {
                        ret = Symbolic.and(ret, curr_condition);
                    }
                }
            }
        }
        return ret;
    }

    /**
    * Checks if there is any induction variable placed at the specified loop
    * and any inner loops that increments the induction variable.
    * @param loop the given loop.
    * @return true if there is such an induction variable, false otherwise.
    */
    private boolean hasInnerInduction(Loop loop) {
        for (int i = 0; i < eligible_loops.indexOf(loop); i++) {
            Loop curr_loop = eligible_loops.get(i);
            if (IRTools.isAncestorOf(loop, curr_loop)) {
                Set<Symbol> curr_ivs =
                        new LinkedHashSet<Symbol>(getIVs(curr_loop));
                curr_ivs.retainAll(loop_to_ivs.get(loop));
                if (!curr_ivs.isEmpty()) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
    * Returns data associated the specified loop.
    * @param loop the given loop.
    * @return the data associated with the loop in string.
    */
    private String getReport(Loop loop) {
        StringBuilder sb = new StringBuilder(80);
        sb.append("#");
        sb.append(eligible_loops.size() - eligible_loops.indexOf(loop));
        sb.append(" ");
        sb.append(LoopTools.toControlString(loop));
        sb.append(" ivs=[");
        if (loop_to_ivs.get(loop) != null) {
            for (Symbol iv : loop_to_ivs.get(loop)) {
                sb.append(iv.getSymbolName());
                if (iv_to_divisible.get(iv) == true) {
                    sb.append("(/)");
                }
                sb.append(", ");
            }
        }
        sb.append("], variants=");
        sb.append(loop_to_variants.get(loop));
        return sb.toString();
    }

    /**
    * Prints out the result of IV analysis for the specified loop recursively.
    * @param loop the given loop.
    * @param indent the current indentation.
    */
    private void reportAnalysis(Loop loop, String indent) {
        if (!eligible_loops.contains(loop)) {
            return;
        }
        PrintTools.printlnStatus(1, indent, getReport(loop));
        indent += "  ";
        DFIterator<Statement> iter =
                new DFIterator<Statement>(loop.getBody(), Statement.class);
        iter.pruneOn(Loop.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            if (stmt instanceof Loop) {
                reportAnalysis((Loop)stmt, indent);
            } else if (loop_to_stmts.get(loop).containsKey(stmt)) {
                PrintTools.printlnStatus(1, tag, indent, stmt, "[ inc =",
                        loop_to_stmts.get(loop).get(stmt), "]");
            }
        }
        if (loop == eligible_loops.getLast()) {
            for (Symbol iv : iv_to_loops.keySet()) {
                PrintTools.printlnStatus(1,tag,"ivs=",iv.getSymbolName(),":");
                Set<Loop> iv_loops = iv_to_loops.get(iv);
                for (Loop iv_loop : iv_loops) {
                    PrintTools.printlnStatus(1, tag, "    ",getReport(iv_loop));
                }
            }
        }
    }

    /**
    * Matches induction statements in the specified loop.
    * @param loop the given loop.
    * @return PASS if matching is successful, FAIL otherwise.
    */
    private int matchInduction(Loop loop) {
        if (!LoopTools.isCanonical(loop)) {
            PrintTools.printlnStatus(1, tag,
                "Analysis stops due to a non-canonical loop:", getReport(loop));
            return FAIL;
        }
        eligible_loops.add(0, loop);
        loop_to_stmts.put(loop, new LinkedHashMap<Statement, Expression>());
        Set<Symbol> variants = new LinkedHashSet<Symbol>();
        loop_to_variants.put(loop, variants);

        Set<Traversable> unsafe_stmts = new LinkedHashSet<Traversable>();
        DFIterator<Statement> iter =
                new DFIterator<Statement>(loop.getBody(), Statement.class);
        iter.pruneOn(Loop.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            // Skip over non-statement IRs and compound statements.
            if (stmt instanceof CompoundStatement) {
                continue;
            }
            // Skip over any IR that belongs to unsafe statements.
            if (containsAncestor(unsafe_stmts, stmt)) {
                continue;
            }
            // From here all loops and statements are eligible for matching.
            if (stmt instanceof Loop) {
                if (stmt instanceof ForLoop) {                           
                    // Adds modified variables in the control
                    // part of the inner loops.
                    ForLoop forloop = (ForLoop)stmt;
                    variants.addAll(DataFlowTools.getDefSymbol(
                            forloop.getInitialStatement()));
                    variants.addAll(DataFlowTools.getDefSymbol(
                            forloop.getCondition()));
                    variants.addAll(DataFlowTools.getDefSymbol(
                            forloop.getStep()));
                }
                if (matchInduction((Loop)stmt) != PASS) {
                    return FAIL;
                }
            } else if (stmt instanceof ExpressionStatement) {
                matchInduction(loop, (ExpressionStatement)stmt);
            } else {
                // Other types of statements are all considered to have side
                // effect, which includes any kinds of conditional branches.
                variants.addAll(DataFlowTools.getDefSymbol(stmt));
                unsafe_stmts.add(stmt);
            }
        }
        // Add locally (loop) defined symbols in the set of variants.
        variants.addAll(SymbolTools.getSymbols((SymbolTable)loop.getBody()));
        return PASS;
    }

    /**
    * Matches induction statements.
    * @param loop the enclosing loop.
    * @param stmt the given statement to be examined.
    */
    private void matchInduction(Loop loop, ExpressionStatement stmt) {
        Expression e = stmt.getExpression();
        if (e != null) {
            if (e instanceof AssignmentExpression) {
                matchAssignment(loop, (AssignmentExpression)e);
            } else if (e instanceof UnaryExpression) {
                matchUnary(loop, (UnaryExpression)e);
            } else {
                loop_to_variants.get(loop).addAll(
                        DataFlowTools.getDefSymbol(e));
            }
        }
    }

    /**
    * Matches assignment induction that has one of the following forms:
    * iv = iv+inc, iv += inc, iv -= inc.
    * @param loop the enclosing loop.
    * @param e the given assignment expression.
    */
    private void matchAssignment(Loop loop, AssignmentExpression e) {
        Statement stmt = (Statement)e.getParent();
        Expression lhs = e.getLHS(), rhs = e.getRHS();
        String op = e.getOperator().toString();
        Symbol var = SymbolTools.getSymbolOf(lhs);
        if (lhs instanceof Identifier && SymbolTools.isInteger(var)) {
            Expression inc = null, zero = new IntegerLiteral(0);
            if (op.equals("=") &&
                !IRTools.containsSymbol(inc=Symbolic.subtract(rhs, lhs), var) ||
                op.equals("+=") &&
                !IRTools.containsSymbol(inc=Symbolic.simplify(rhs), var) ||
                op.equals("-=") &&
                !IRTools.containsSymbol(inc=Symbolic.subtract(zero, rhs),var)) {
                ivstmt_to_inc.put(stmt, inc);
                Set<Symbol> vars_in_inc = SymbolTools.getAccessedSymbols(inc);
                inc = RangeAnalysis.query(stmt).substituteForward(
                        inc, vars_in_inc);
                loop_to_stmts.get(loop).put(stmt, inc);
                PrintTools.printlnStatus(3,
                        tag, "Found", stmt, "in", getReport(loop));
            } else {
                loop_to_variants.get(loop).addAll(
                        DataFlowTools.getDefSymbol(e));
            }
        } else {
            loop_to_variants.get(loop).addAll(DataFlowTools.getDefSymbol(e));
        }
    }

    /**
    * Matches unary induction that has one of the following forms:
    * iv++, iv--, ++iv, --iv.
    * @param loop the enclosing loop.
    * @param e the given unary expression.
    */
    private void matchUnary(Loop loop, UnaryExpression e) {
        Statement stmt = e.getStatement();
        Expression ue = e.getExpression();
        String op = e.getOperator().toString();
        Symbol var = SymbolTools.getSymbolOf(ue);
        if (ue instanceof Identifier && SymbolTools.isInteger(var) &&
            (op.equals("++") || op.equals("--"))) {
                Expression inc = new IntegerLiteral(1);
                if (op.equals("--")) {
                    inc = new IntegerLiteral(-1);
                }
                ivstmt_to_inc.put(stmt, inc);
                loop_to_stmts.get(loop).put(stmt, inc);
                PrintTools.printlnStatus(3,
                        tag, "Found", stmt, "in", getReport(loop));
        } else {
            loop_to_variants.get(loop).addAll(DataFlowTools.getDefSymbol(e));
        }
    }

    /**
    * Returns induction variables from the given induction statement.
    * @param stmt the given induction statement.
    * @return the symbol of the induction variable.
    */
    private Symbol getIV(Statement stmt) {
        Expression e = (Expression)stmt.getChildren().get(0);
        if (e instanceof AssignmentExpression) {
            return SymbolTools.getSymbolOf(((AssignmentExpression)e).getLHS());
        } else if (e instanceof UnaryExpression) {
            return SymbolTools.getSymbolOf(
                    ((UnaryExpression)e).getExpression());
        } else {
            throw new InternalError(tag + " Unsupported induction statement");
        }
    }

    /**
    * Returns a set of induction variables in induction statements that belong
    * to the specified loop.
    * @param loop the given loop.
    * @return the set of induction variables.
    */
    private Set<Symbol> getIVs(Loop loop) {
        Set<Symbol> ret = new LinkedHashSet<Symbol>();
        Map<Statement, Expression> curr_incs = loop_to_stmts.get(loop);
        for (Statement stmt : curr_incs.keySet()) {
            ret.add(getIV(stmt));
        }
        return ret;
    }

    /**
    * Returns an substitution order of the induction variables for the specified
    * loop.
    * @param loop the given loop.
    * @return the ordered list.
    */
    private List<Symbol> getSubstOrder(Loop loop) {
        List<Symbol> ret = new LinkedList<Symbol>();
        Set<Symbol> ivs = loop_to_ivs.get(loop);
        if (ivs.isEmpty()) {
            return ret;
        }
        DFAGraph iv_graph = new DFAGraph();
        for (Symbol iv : ivs) {
            iv_graph.addNode(new DFANode("iv", iv));
        }
        for (Symbol iv : ivs) {
            Set<Loop> iv_loops = iv_to_loops.get(iv);
            DFANode iv_node = iv_graph.getNodeWith("iv", iv);
            for (Loop iv_loop : iv_loops) {
                Map<Statement, Expression> iv_stmts =
                        loop_to_stmts.get(iv_loop);
                for (Statement stmt : iv_stmts.keySet()) {
                    if (getIV(stmt) != iv) {
                        continue;
                    }
                    Expression inc = iv_stmts.get(stmt);
                    Set<Symbol> accessed = SymbolTools.getAccessedSymbols(inc);
                    for (Symbol acc : accessed) {
                        DFANode acc_node = iv_graph.getNodeWith("iv", acc);
                        if (acc_node != null) {
                            iv_graph.addEdge(iv_node, acc_node);
                            // dependece order
                        }
                    }
                }
            }
        }
        for (DFANode root : iv_graph.getEntryNodes()) {
            for (Object component : iv_graph.getSCC(root)) {
                if (component instanceof List &&
                        ((List)component).size() == 1) {
                    DFANode iv_node = (DFANode)((List)component).get(0);
                    Symbol iv_symbol = (Symbol)iv_node.getData("iv");
                    if (!ret.contains(iv_symbol)) {
                        ret.add(iv_symbol);
                    }
                } else {
                    PrintTools.printlnStatus(3, tag,
                            "Found cyclic induction variables");
                    ret.clear();
                    break;
                }
            }
        }
        return ret;
    }

    /**
    * Performs induction variable substitution in post order in the loop nest.
    * @param the outer-most loop.
    * @return PASS.
    */
    private int transform(Loop loop) {
        PrintTools.printlnStatus(1, tag, "Transformation starts...");
        // Decide whether the loop nest needs runtime test.
        Statement orig = null;
        Expression nzt_condition = null;
        if (hasInnerInduction(loop)) {
            nzt_condition = getNZTConditions(loop);
            if (nzt_condition != null) {
                int nzt_eval = RangeAnalysis.query((Statement)loop).
                        evaluateLogic(nzt_condition);
                if (nzt_eval == 1) {
                    // found to be true at the outer-most loop.
                    nzt_condition = null;
                }
            }
        }
        if (nzt_condition != null) {
            orig = ((Statement)loop).clone();
        }
        // After-loop increments for the outermost loop needs to be cached to be
        // placed properly in the multi-version loop.
        List<Statement> live_increments  = new LinkedList<Statement>();
        // Substitution in the original IR in post order.
        for (int i = 0; i < eligible_loops.size(); ++i ) {
            Loop curr_loop = eligible_loops.get(i);
            List<Symbol> ivs = getSubstOrder(curr_loop);
            Set<Symbol> failed_ivs = new LinkedHashSet<Symbol>();
            for (Symbol iv : ivs) {
                inc_after_stmt.clear();
                inc_into_loop.clear();
                Expression init = getInitialValue(curr_loop, iv);
                if (init == null) {
                    // initial value not available; use iv itself.
                    init = new Identifier(iv);
                }
                Expression inc = findIncrement(curr_loop, iv);
                if (inc == null) {
                    failed_ivs.add(iv);
                    continue;
                }
                if (replace(curr_loop, iv, init) == PASS) {
                    PrintTools.printlnStatus(1, tag, "Successfully replaced",
                            iv.getSymbolName());
                }
                // Increments after the loop (TODO: if live)
                Statement stmt = (Statement)curr_loop;
                Expression assign = new AssignmentExpression(new Identifier(iv),
                        AssignmentOperator.ADD, inc);
                Statement assign_stmt = new ExpressionStatement(assign);
                // The only case where conditional last value is necessary -
                // inner loops have either nz-trip count or no general
                // induction, and the current loop is not nz-trip loop. This
                // condition was computed before as nzt_condition.
                Expression curr_nzt = getNZTCondition(curr_loop);
                if (nzt_condition == null && curr_nzt != null) {
                    assign_stmt = new IfStatement(curr_nzt, assign_stmt);
                }
                CompoundStatement parent = (CompoundStatement)stmt.getParent();
                parent.addStatementAfter(stmt, assign_stmt);
                if (curr_loop == loop) {
                    live_increments.add(assign_stmt);
                }
            }
            // Remove induction statements - another option is to replace it
            // with closed form expression(see replace method).
            for (int j = 0; j <= i; ++j) {
                Set<Statement> iv_stmts = new LinkedHashSet<Statement>(
                        loop_to_stmts.get(eligible_loops.get(j)).keySet());
                for (Statement iv_stmt : iv_stmts) {
                    Symbol iv = getIV(iv_stmt);
                    if (ivs.contains(iv) && !failed_ivs.contains(iv)) {
                        loop_to_stmts.get(curr_loop).remove(iv_stmt);
                        iv_stmt.getParent().removeChild(iv_stmt);  
                    }
                }
            }
        }
        // Multi-version loop generation.
        if (nzt_condition != null) {
            Statement transformed = ((Statement)loop).clone();
            Statement if_stmt = new IfStatement(
                    nzt_condition, transformed, orig);
            CompoundStatement parent = (CompoundStatement)loop.getParent();
            parent.addStatementAfter((Statement)loop, if_stmt);
            parent.removeChild(loop);
            CompoundStatement true_parent =
                    (CompoundStatement)transformed.getParent();
            for (Statement stmt : live_increments) {
                parent.removeChild(stmt);
                true_parent.addStatementAfter(transformed, stmt);
            }
        }
        return PASS;
    }

    /**
    * Finds increments into the loop and returns the total increments after
    * the given loop. The total increments after each inner loop and each
    * induction statment are also computed and stored in the "inc_after" map.
    * It is important to know that "inc_after" stores cumulative increments
    * since the entry of the specified loop.
    * @param loop the given loop.
    * @param iv the induction variable.
    * @return the total increment of iv after the given loop.
    */
    private Expression findIncrement(Loop loop, Symbol iv) {
        Identifier id = (Identifier)loop_to_bounds.get(loop).get(0);
        Expression lb = loop_to_bounds.get(loop).get(1);
        Expression ub = loop_to_bounds.get(loop).get(2);
        Expression inc = new IntegerLiteral(0);
        DFIterator<Statement> iter =
                new DFIterator<Statement>(loop.getBody(), Statement.class);
        iter.pruneOn(Loop.class);
        iter.pruneOn(ExpressionStatement.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            Expression stmt_inc = null;
            if (stmt instanceof Loop) {
                // Consider only the loops relevant to the induction variable.
                if (eligible_loops.contains(stmt)) {
                    Expression inner_inc = findIncrement((Loop)stmt, iv);
                    if (inner_inc == null) {
                        return null;
                    }
                    inc = Symbolic.add(inc, inner_inc);
                    if (iv_to_divisible.get(iv)) {
                        inc = Symbolic.simplifyDivisible(inc);
                    }
                    PrintTools.printlnStatus(3, tag, "inc_after [",
                            getReport((Loop)stmt), "] =", inc);
                    inc_after_stmt.put(stmt, inc);
                }
            } else if ((stmt_inc=loop_to_stmts.get(loop).get(stmt)) != null &&
                       getIV(stmt) == iv) {
                inc = Symbolic.add(inc, stmt_inc);
                if (iv_to_divisible.get(iv)) {
                    inc = Symbolic.simplifyDivisible(inc);
                }
                PrintTools.printlnStatus(3, tag, "inc_after [",stmt,"] =", inc);
                inc_after_stmt.put(stmt, inc);
            }
        }
        Boolean iv_is_divisible = iv_to_divisible.get(iv);
        // Closed-form computation at the start of current iteration.
        Expression prev_ub = Symbolic.subtract(id, new IntegerLiteral(1));
        Expression sum = Symbolic.getClosedFormSum(
                id, lb, prev_ub, inc, iv_is_divisible);
        if (sum == null) {
            return null; // Failed closed-form sum computation - no replacement.
        }
        if (iv_is_divisible) {
            sum = Symbolic.simplifyDivisible(sum);
        }
        inc_into_loop.put(loop, sum);
        PrintTools.printlnStatus(3,tag,"inc_into [",getReport(loop),"] =",sum);
        // Closed-form computation after the loop.
        sum = Symbolic.getClosedFormSum(id, lb, ub, inc, iv_is_divisible);
        if (sum == null) {
            return null; // Failed closed-form sum computation - no replacement.
        }
        if (iv_is_divisible) {
            sum = Symbolic.simplifyDivisible(sum);
        }
        return sum;
    }

    /**
    * Checks if the given set of IRs contains an ancestor of the specified 
    * traversable.
    * @param ancs the set of traversable objects.
    * @param t the specified traversable.
    */
    private static boolean
            containsAncestor(Set<? extends Traversable> ancs, Traversable t) {
        for (Traversable anc : ancs) {
            if (IRTools.isAncestorOf(anc, t)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Visits each loop and replaces the iv with its closed-form expression.
    * Any descendants, including inner loops, that are not part of induction
    * variable computation will be just considered as a traversable object
    * where replacement occurs.
    * @param loop the given loop.
    * @param iv the given induction variable.
    * @param init the initial value of the induction variable.
    * @return PASS.
    */
    private int replace(Loop loop, Symbol iv, Expression init) {
    // Avoid over-replacement due to depth-first-iteration.
    Set<Traversable> replaced = new LinkedHashSet<Traversable>();
    // Safety net for some weird assignment to integer variable from floating
    // point numbers.
    List init_type = SymbolTools.getExpressionType(init);
        if (init_type == null) {
            return FAIL;
        }
        if (!SymbolTools.isInteger(init_type)) {
            init = new Typecast(iv.getTypeSpecifiers(), init.clone());
        }
        Expression init_val = Symbolic.add(init, inc_into_loop.get(loop));
        if (iv_to_divisible.get(iv)) {
            init_val = Symbolic.simplifyDivisible(init_val);
        }
        Expression val = init_val;
        DFIterator<Statement> iter =
                new DFIterator<Statement>(loop.getBody(), Statement.class);
        iter.pruneOn(Loop.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            Expression inc = null;
            if (stmt instanceof CompoundStatement ||
                containsAncestor(replaced, stmt)) {
                continue; // Skip compound stmts and stmt already replaced.
            }
            if (stmt instanceof Loop) {
        //Loop inner = (Loop)o;
                // Loops that are not part of iv computation are just modified
                // using the current value increments.
                if (loop_to_stmts.get(stmt) == null) {
                    IRTools.replaceSymbolIn(stmt, iv, val);
                    replaced.add(stmt);
                } else {
                    // Loops that are part of iv computation.
                    replace((Loop)stmt, iv, val);
                    val = Symbolic.add(init_val, inc_after_stmt.get(stmt));
                    if (iv_to_divisible.get(iv)) {
                        val = Symbolic.simplifyDivisible(val);
                    }
                }
            } else if ((inc=loop_to_stmts.get(loop).get(stmt)) != null) {
                ExpressionStatement iv_stmt = (ExpressionStatement)stmt;
                if (getIV(iv_stmt) == iv) {
                    // induction statements for iv.
                    val = Symbolic.add(init_val, inc_after_stmt.get(iv_stmt));
                    if (iv_to_divisible.get(iv)) {
                        val = Symbolic.simplifyDivisible(val);
                    }
                    // Replace induction statement - unnecessary if it is
                    // removed later.
                    // AssignmentExpression iv_expr = iv_stmt.getExpression();
                    // iv_expr.setOperator(AssignmentOperator.NORMAL);
                    // iv_expr.getRHS().swapWith((Expression)val.clone());
                } else {
                    // other induction statements.
                    IRTools.replaceSymbolIn(stmt, iv, val);
                    replaced.add(stmt);
                }
                // Update the increment cache as well.
                loop_to_stmts.get(loop).put(iv_stmt,
                        IRTools.replaceSymbol(inc, iv, val));
            } else {
                IRTools.replaceSymbolIn(stmt, iv, val);
                replaced.add(stmt);
            }
        }
        return PASS;
    }

    /**
    * Returns the initial value of the specified induction variable before
    * entering the given loop.
    */
    private Expression getInitialValue(Loop loop, Symbol iv) {
        List<Traversable> siblings = loop.getParent().getChildren();
        for (int i = siblings.indexOf(loop)-1; i >= 0; i--) {
            Traversable elder = siblings.get(i);
            if (IRTools.containsClass(elder, FunctionCall.class)) {
                return null;
            }
            if (DataFlowTools.getDefSymbol(elder).contains(iv)) {
                if (!(elder instanceof ExpressionStatement)) {
                    return null;
                }
                Expression expr = ((ExpressionStatement)elder).getExpression();
                if (!(expr instanceof AssignmentExpression)) {
                    return null;
                }
                AssignmentExpression assign_expr = (AssignmentExpression)expr;
                if (assign_expr.getOperator() != AssignmentOperator.NORMAL) {
                    return null;
                }
                return assign_expr.getRHS();
            }
        }
        return null;
    }

    // Conditions for a variable to be a wrap-around variable.
    //   1. It has a symbolic constant range(lb==ub) before the loop.
    //   2. There exist an assignment statement to the variable with
    //      non-loop-variant expressions.
    //   3. The statement in 2 should be a dominant node from the exit node
    //      of the loop in the reverse graph of the CFG.
    private boolean isWrapAroundVariable(Loop loop, Symbol var) {
        Set<Symbol> variants = loop_to_variants.get(loop);
        // Here we use adjusted set of loop-variants from the previous analysis
        // which does not count induction statements or loop indices
        // as loop-variant variables, so detection of destructive assignment to
        // the given variable should consider that fact.
        RangeDomain before_rd = RangeAnalysis.query((Statement)loop);
        Expression before_range = before_rd.getRange(var);
        Map<Statement, Expression> iv_stmts = loop_to_stmts.get(loop);
        // Condition#1
        if (before_range == null ||
            before_range instanceof RangeExpression ||
            IRTools.containsSymbols(before_range, variants)) {
            return false;
        }
        // Condition#2 and Condition#3
        Statement last_value_stmt = null;
        List<Traversable> stmts_in_loop = loop.getBody().getChildren();
        for (int i = stmts_in_loop.size()-1; i >= 0; i--) {
            Traversable t = stmts_in_loop.get(i);
            if (t instanceof ExpressionStatement &&
                t.getChildren().get(0) instanceof AssignmentExpression) {
                AssignmentExpression assign =
                        (AssignmentExpression)t.getChildren().get(0);
                if (SymbolTools.getSymbolOf(assign.getLHS()) == var &&
                    !IRTools.containsSymbols(assign.getRHS(), variants) &&
                    (iv_stmts == null || !iv_stmts.containsKey(t))) {
                    last_value_stmt = (Statement)t;
                    Set<Statement> wrap_stmts = loop_to_wrap_stmts.get(loop);
                    if (wrap_stmts == null) {
                        wrap_stmts = new LinkedHashSet<Statement>();
                        loop_to_wrap_stmts.put(loop, wrap_stmts);
                    }
                    wrap_stmts.add(last_value_stmt);
                    PrintTools.printlnStatus(3, tag, "Found wrap-around",
                            last_value_stmt, "in", getReport(loop));
                    break;
                }
            } else {
                break;
            }
        }
        return (last_value_stmt != null);
    }

    // Performs loop-peeling on any qualified loops.
    private void removeWrapAroundVariables() {
        // 1. Assign index = first value;
        // 2. Insert the code section for the first iteration.
        // 3. Move the wrap-around statement at the start of the loop body.
        //    and fix the loop bounds.
        // 4. Insert last value assignment after the loop (with condition).
        for (Loop loop : eligible_loops) {
            Set<Statement> wrap_stmts = loop_to_wrap_stmts.get(loop);
            if (wrap_stmts == null) {
                continue;
            }
            ForLoop for_loop = (ForLoop)loop;
            CompoundStatement parent = (CompoundStatement)for_loop.getParent();
            // 1. Assign index = first value;
            Statement init_stmt = for_loop.getInitialStatement();
            parent.addStatementBefore(for_loop, init_stmt.clone());
            // 2. Insert the code section for the first iteration.
            for (Traversable child : for_loop.getBody().getChildren()) {
                if (!wrap_stmts.contains(child)) {
                    parent.addStatementBefore(for_loop,
                            ((Statement)child).clone());
                }
            }
            // 3. Move the wrap-around statement at the start of the loop body.
            Identifier index = (Identifier)loop_to_bounds.get(loop).get(0);
            Expression step = LoopTools.getIncrementExpression(loop);
            CompoundStatement body = (CompoundStatement)for_loop.getBody();
            Statement first_stmt = (Statement)body.getChildren().get(0);
            for (Statement wrap_stmt : wrap_stmts) {
                body.removeChild(wrap_stmt);
                Statement moved_stmt = wrap_stmt.clone();
                IRTools.replaceSymbolIn(moved_stmt, index.getSymbol(),
                        Symbolic.subtract(index, step));
                ExpressionStatement expr_stmt = (ExpressionStatement)moved_stmt;
                expr_stmt.setChild(0,
                        Symbolic.simplify(expr_stmt.getExpression()));
                body.addStatementBefore(first_stmt, moved_stmt);
            }
            // Fix the initial statement.
            Expression init_expr = (Expression)init_stmt.getChildren().get(0);
            Expression rhs = (Expression)init_expr.getChildren().get(1);
            init_expr.setChild(1, Symbolic.add(rhs, step));
            // 4. Insert last value assignment after the loop (with condition).
            Expression ub = loop_to_bounds.get(loop).get(2);
            for (Statement wrap_stmt : wrap_stmts) {
                IRTools.replaceSymbolIn(wrap_stmt, index.getSymbol(), ub);
            }
            Expression nzt_condition = getNZTCondition(for_loop);
            if (nzt_condition == null) {
                // no need for runtime checking.
                for (Statement wrap_stmt : wrap_stmts) {
                    parent.addStatementAfter(for_loop, wrap_stmt);
                }
            } else {
                CompoundStatement true_body = new CompoundStatement();
                for (Statement wrap_stmt : wrap_stmts) {
                    true_body.addStatement(wrap_stmt);
                }
                IfStatement if_stmt = new IfStatement(nzt_condition, true_body);
                parent.addStatementAfter(for_loop, if_stmt);
            }
        }
    }
}
