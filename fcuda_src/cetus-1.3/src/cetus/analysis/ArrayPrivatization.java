package cetus.analysis;

import cetus.exec.Driver;
import cetus.hir.*;

import java.util.*;

/**
 * ArrayPrivatization performs privatization analysis of the program. It tries
 * to find privatizable variables (scalars and arrays) which are written first
 * then read in a loop body.
 * The high-level algorithm in below describes the process of detecting
 * privatizable variables, both scalars and array sections, in a loop. The set
 * operations that appear in the algorithm are performed on the array sections
 * if the variable is an array. We use the power of symbolic analysis
 * techniques in Cetus to make the symbolic section operation possible. For
 * example, <code>[1:m] (intersect) [1:n]</code> results in <code>[1:n]</code>
 * if the expression comparison tool with the value range set can decide
 * <code>n</code> is less than or equal to <code>m</code>.
 * <p>
 * The algorithm traverses a loop nest from the innermost to the outermost loop.
 * At each level, it first collects <code>definitions</code> (write references)
 * and <code>uses</code> (read references) in the loop body. Uses that are
 * covered by prior definitions create privatizable variables (or array
 * sections) for the current loop. The other uses are
 * <code>upward exposed</code>, creating read
 * references seen in the outer loop. Second, the algorithm aggregates all
 * these array sections over the loop iteration space, creating the array
 * sections that are private, written and upward exposed for the entire loop.
 * The aggregation for the written sections (<code>DEF</code>) computes the
 * must-defined regular sections of the arrays over the entire loop iteration
 * while the aggregation of upward-exposed sections (<code>UEU</code>) requires
 * only the conservative value ranges of the sections (may-used sections). This
 * algorithm is a slightly simpler version of the one used in the Polaris
 * parallelizing compiler for Fortran77 programs.
 * <pre>
 * procedure Privatization(L)
 *
 *   Input : Loop L
 *   Output: DEF[L], UEU[L], PRI[L]
 *   // DEF: Definded set
 *   // USE: Used set
 *   // UEU: Upward-exposed set
 *   // PRI: Private variables
 *   // (^): intersection, (v): union
 *
 *   foreach direct inner loop L' in L
 *     (DEF[L'],USE[L']) = Privatization(L')
 *
 *   G(N,E) = BuildCFG(L)
 *   // BuildCFG builds a CFG with the inner loops represented as super nodes
 *
 *   Iteratively solve data flow equation DEF for node n in N
 *     DEFin[n] = (^) {DEFout[p], p in predecessors of n}
 *     DEFout[n] = (DEFin[n]-KILL[n]) (v) DEF[n]
 *
 *   DEF[L] = {}
 *   UEU[L] = {}
 *   PRI[L] = CollectCandidates(L)
 *   // CollectCandidates collects variables that satisfies the following:
 *   // 1. It has write access in the loop body.
 *   // 2. Its subscript do not contain loop-variant variables.
 *
 *   foreach loop-exiting node e in N
 *     DEF[L] = DEF[L] (^) DEFout[e]
 *
 *   foreach node n in N
 *     UEU[L] = UEU[L] (v) (USE[n]-DEFin[n])
 *
 *   foreach variable v in UEU[L]
 *     PRI[L] = PRI[L]-{v}
 *
 *   DEF[L] = AggregateDEF(DEF[L])
 *   UEU[L] = AggregateUSE(UEU[L])
 *   // AggregateDEF aggregates array sections in DEF (MUST set)
 *   // AggregateUSE aggregates array sections in USE (MAY set)
 *
 *   return (DEF[L], UEU[L])
 *
 * end
 * </pre>
 * Array privatization is invoked by specifying the flag
 * <code>-privatize</code>,
 * and the result is stored as an annotation that contains the set of private
 * variables. We do not consider a variable with user-defined type as a
 * candidate private variable, but intend to widen the scope of the analysis
 * in the future.
 */
public class ArrayPrivatization extends AnalysisPass {
    // Map: inner loop => DEF set
    private Map<Loop, Section.MAP> def_map;

    // Map: inner loop => USE set
    private Map<Loop, Section.MAP> use_map;

    // Map: loop => private variables 
    private Map<Loop, Set<Symbol>> pri_map;

    // Map: loop => may private in outer loop
    private Map<Loop, Section.MAP> may_map;

    // CFGraph for live variable analysis
    private CFGraph live_cfg;

    // Current loop
    private Loop current_loop;

    // Loop-variant variables for the current loop
    private Set<Symbol> loop_variants;

    // Loop-exit range domain for the current loop
    private RangeDomain exit_range;

    // Loop-exit induction increments for the current loop
    private Map<Symbol, Expression> exit_ind;

    // Whole program's alias information.
    private AliasAnalysis alias_analysis;

    // Alias result.
    private int alias_result = 0;

    // Debug
    private int debug;

    // Pass name
    private static final String tag = "[ArrayPrivatization]";

    /**
    * Constructs an array privatization analyzer for a program.
    *
    * @param program the input program.
    */
    public ArrayPrivatization(Program program) {
        super(program);
        debug = PrintTools.getVerbosity();
        // Call alias analysis.
        alias_analysis = new AliasAnalysis(program);
        alias_analysis.start();
    }

    /**
    * Returns the pass name.
    *
    * @return the pass name in string.
    */
    public String getPassName() {
        return tag;
    }

    /**
    * Starts array privatization by calling the procedure-level driver for each
    * procedure within the program.
    */
    public void start() {
        DFIterator<Procedure> iter =
                new DFIterator<Procedure>(program, Procedure.class);
        iter.pruneOn(Procedure.class);
        while (iter.hasNext()) {
            Procedure p = iter.next();
            if (alias_result != 0) {
                PrintTools.printlnStatus(0, tag,
                    "[WARNING] Privatization stops due to all-to-all alias");
                break;
            }
            analyzeProcedure(p);
        }
    }

    /**
    * Starts array privatization for a procedure. It first computes the range
    * map of the procedure to enable subsequent symbolic operations, then
    * calls {@link #analyzeLoop(Loop)} to analyze the outer-most loops in the
    * procedure. The final step is to annotate the analysis result for each
    * loop in the procedure.
    *
    * @param proc the input procedure.
    */
    public void analyzeProcedure(Procedure proc) {
        double timer = Tools.getTime();
        PrintTools.printlnStatus(1, tag, "Procedure:", proc.getName());
        def_map = new HashMap<Loop, Section.MAP>();
        use_map = new HashMap<Loop, Section.MAP>();
        pri_map = new HashMap<Loop, Set<Symbol>>();
        may_map = new HashMap<Loop, Section.MAP>();
        DFIterator<Loop> iter = new DFIterator<Loop>(proc, Loop.class);
        iter.pruneOn(Loop.class);
        iter.pruneOn(StatementExpression.class); // We do not support this.
        while (iter.hasNext()) {
            analyzeLoop(iter.next());
        }
        addAnnotation(proc);
        PrintTools.printlnStatus(1, tag,
                String.format("...... %.2f seconds", Tools.getTime(timer)));
    }

    /**
    * Creates a cetus private annotation for each loop and inserts it before
    * the loop.
    */
    private void addAnnotation(Procedure proc) {
        // Union private variables for all loops while checking the alias
        // information
        Set<Symbol> pri_set = new LinkedHashSet<Symbol>();
        for (Loop loop : pri_map.keySet()) {
            Set<Symbol> loop_pri_set = pri_map.get(loop);
            if (loop_pri_set == null) {
                continue;
            }
            // Use the result of alias analysis
            Set<Symbol> keys = new LinkedHashSet<Symbol>(loop_pri_set);
            for (Symbol var : keys) {
                Set alias_set =
                        alias_analysis.get_alias_set((Statement)loop, var);
                if (alias_set == null) {
                    continue;
                }
                for (Object aliased : alias_set) {
                    if (aliased.equals("*")) {
                        alias_result = 1;
                        return; // all aliased; stop further analysis.
                    } else if (var != aliased) {
                        loop_pri_set.remove(var);
                        PrintTools.printlnStatus(1, tag,
                                "Removing aliased variable:",var,"=>",aliased);
                        break;  // this variable has a non-empty alias set.
                    }
                }
            }
            pri_set.addAll(loop_pri_set);
        }
        // Get live variables
        CFGraph live_cfg = getLiveVariables(proc, pri_set);
        for (Loop loop : pri_map.keySet()) {
            Statement loop_stmt = (Statement)loop;
            Set<Symbol> loop_pri_set = pri_map.get(loop);
            // Do not add annotation for non-for-loops.
            if (loop_pri_set == null || loop_pri_set.isEmpty()) {
                continue;
            }
            DFANode loop_exit = live_cfg.getNode("stmt-exit", loop);
            Set<Symbol> loop_live_set = loop_exit.getData("live-out");
            if (loop_live_set == null) { // Node that is not reachable backwards
                loop_live_set = new LinkedHashSet<Symbol>();
            } else {
                loop_live_set = new LinkedHashSet<Symbol>(loop_live_set);
            }
            loop_live_set.retainAll(loop_pri_set);
            loop_pri_set.removeAll(loop_live_set);
            loop_live_set.removeAll(loop_pri_set);
            // Move global variables from private => lastprivate.
            // The live variable analysis is not interprocedural.
            for (Symbol var : new LinkedHashSet<Symbol>(loop_pri_set)) {
                if (SymbolTools.isGlobal(var, loop_stmt)) {
                    loop_pri_set.remove(var);
                    loop_live_set.add(var);
                }
            }
            if (loop instanceof ForLoop) { // no need for annotating while loops
                CetusAnnotation note;
                if (!loop_pri_set.isEmpty()) {
                    note = new CetusAnnotation();
                    note.put("private", loop_pri_set);
                    loop_stmt.annotateBefore(note);
                }
                if (!loop_live_set.isEmpty()) {
                    note = new CetusAnnotation();
                    note.put("lastprivate", loop_live_set);
                    loop_stmt.annotateBefore(note);
                }
            }
        }
    }

    /**
    * Analyzes a loop.
    */
    private void analyzeLoop(Loop loop) {
        // -------------------------
        // 1. Analyze the inner loop
        // -------------------------
        DFIterator<Loop> iter = new DFIterator<Loop>(loop.getBody(),Loop.class);
        iter.pruneOn(Loop.class);
        iter.pruneOn(StatementExpression.class);
        while (iter.hasNext()) {
            analyzeLoop(iter.next());
        }
        // Prepare for the analysis; fetch environments, initialization.
        current_loop = loop;
        loop_variants = DataFlowTools.getDefSymbol(current_loop);
        pri_map.put(loop, new LinkedHashSet<Symbol>());
        PrintTools.printlnStatus(2, PrintTools.line_sep, tag, "Loop:",
                LoopTools.toControlString(loop));
        // ------------------------------
        // 2. Build CFG for the loop body
        // ------------------------------
        CFGraph g = buildLoopGraph();
        // -----------------------------------------
        // 3. Solve reaching definition at each node
        // -----------------------------------------
        computeReachingDef(g);
        // Detect loop inductions; it is used when aggregating defined sections.
        computeLoopInduction(g);
        // ----------------------------
        // 4. Compute DEF, UEU, and PRI
        // ----------------------------
        // Summarize the DEF set of the loop
        def_map.put(loop, getDefSet(g));
        // Summarize the USE set of the loop
        use_map.put(loop, getUseSet(g));
        // Add USE from initialized declarations
        addUseFromDeclaration();
        // Collect private variables
        collectPrivateSet(g);
        // Aggregate DEF set and USE set.
        aggregateDef();
        aggregateUse();
    }

    /**
    * Builds a CFG for the loop body; first it includes all control parts such
    * as loop conditions and increments, then remove cycles to adjust the graph
    * for the loop body analysis. It could be easier to create the CFG of the
    * compound statement (loop body) but we also need to visit the nodes that
    * control the loop for more general analysis of the loop.
    */
    private CFGraph buildLoopGraph() {
        CFGraph ret = new CFGraph((Traversable)current_loop, Loop.class);
        if (current_loop instanceof DoLoop) {
            DFANode do_node = ret.getNode("stmt", current_loop);
            DFANode condition_node = do_node.getData("do-condition");
            DFANode back_to_node = condition_node.getData("true");
            back_to_node.putData("header", current_loop);
            ret.removeEdge(condition_node, back_to_node);
            ret.removeNode((DFANode)do_node.getData("do-exit"));
        } else if (current_loop instanceof WhileLoop) {
            DFANode while_node = ret.getNode("stmt", current_loop);
            DFANode condition_node = while_node.getData("while-condition");
            condition_node.putData("header", current_loop);
            Set<DFANode> back_from_nodes =
                    new LinkedHashSet<DFANode>(condition_node.getPreds());
            for (DFANode back_from_node : back_from_nodes) {
                if (back_from_node.getData("stmt") != current_loop) {
                    ret.removeEdge(back_from_node, condition_node);
                }
            }
            ret.removeNode((DFANode)while_node.getData("while-exit"));
        } else if (current_loop instanceof ForLoop) {
            DFANode for_node = ret.getNode("stmt", current_loop);
            DFANode step_node = for_node.getData("for-step");
            DFANode condition_node = for_node.getData("for-condition");
            condition_node.putData("header", current_loop);
            ret.removeEdge(step_node, condition_node);
            ret.removeNode((DFANode)for_node.getData("for-exit"));
        }
        // Remove any unreachable subgraphs.
        ret.topologicalSort(ret.getNodeWith("stmt", "ENTRY"));
        List<DFANode> removable = new ArrayList<DFANode>();
        Iterator<DFANode> iter = ret.iterator();
        while (iter.hasNext()) {
            DFANode curr = iter.next();
            if ((Integer)curr.getData("top-order") < 0) {
                removable.add(curr);
            }
        }
        for (DFANode curr : removable) {
            ret.removeNode(curr);
        }
        // Add RangeDomain to each nodes.
        getRangeDomain(ret);
        if (debug >= 3) {
            PrintTools.printlnStatus(ret.toDot("range,ir,tag", 2), 3);
        }
        return ret;
    }

    /**
    * Collects USE from the initialized declarations.
    */
    private void addUseFromDeclaration() {
        Section.MAP new_use_map = use_map.get(current_loop);
        RangeDomain rd = RangeAnalysis.query((Statement)current_loop);
        if (rd == null) {
            rd = new RangeDomain();
        } else {
            rd = rd.clone();
        }
        DFIterator<Initializer> iter = new DFIterator<Initializer>(
                current_loop.getBody(), Initializer.class);
        iter.pruneOn(Loop.class);
        while (iter.hasNext()) {
            List<Traversable> children = iter.next().getChildren();
            for (int i = 0; i < children.size(); i++) {
                for (Expression use : DataFlowTools.getUseSet(children.get(i))){
                    new_use_map = new_use_map.unionWith(
                            getSectionMap(use, false), rd);
                }
            }
        }
        use_map.put(current_loop, new_use_map);
    }

    /**
    * Computes the reaching definition for each node in the CFG.
    */
    private void computeReachingDef(CFGraph g) {
        TreeMap<Integer, DFANode> work_list = new TreeMap<Integer, DFANode>();
        // Enter the entry node in the worklist.
        DFANode entry = g.getNodeWith("stmt", "ENTRY");
        entry.putData("def-in", new Section.MAP());
        work_list.put((Integer)entry.getData("top-order"), entry);
        // Do iterative steps
        while (!work_list.isEmpty()) {
            DFANode node = work_list.remove(work_list.firstKey());
            Section.MAP curr_map = null;
            for (DFANode pred : node.getPreds()) {
                Section.MAP pred_map = pred.getData("def-out");
                if (curr_map == null) {
                    curr_map = pred_map.clone();
                } else {
                    curr_map = curr_map.intersectWith(
                            pred_map, (RangeDomain)node.getData("range"));
                }
            }
            Section.MAP prev_map = node.getData("def-in");
            if (prev_map == null || !prev_map.equals(curr_map)) {
                node.putData("def-in", curr_map);
                // Handles data kill, union, etc.
                computeOutDef(node);
                for (DFANode succ : node.getSuccs()) {
                    work_list.put((Integer)succ.getData("top-order"), succ);
                }
            }
        }
    }

    /**
    * Computes DEFout from DEFin for the node while collecting candidate private
    * variables; if a defined variable is a scalar or an array with
    * loop-invariant subscript expression, it is a candidate.
    * If the node contains any function calls, the following conservative
    * decision is made:
    *   Every global variables and actual parameters of which addresses are
    *   taken are considered as being modified.
    *
    * This decision affects the analysis in two aspects:
    * 1) Array section containing the above variables should be killed.
    * 2) Used array sections containing the above variables are overly
    *    approximated and we handle this case by removing the DEF array section
    *    entries for the variables
    */
    private void computeOutDef(DFANode node) {
        Section.MAP in = new Section.MAP(), out = null;
        RangeDomain rd = node.getData("range");
        Set<Symbol> killed_vars = new LinkedHashSet<Symbol>();
        Set<Symbol> pri_set = pri_map.get(current_loop);
        if (node.getData("def-in") != null) {
            in = ((Section.MAP)node.getData("def-in")).clone();
        }
        if (node.getData("super-entry") != null) {
            Statement loop = node.getData("super-entry");
            out = def_map.get(loop);
            killed_vars.addAll(DataFlowTools.getDefSymbol(loop));
            // Add may private variables in the candidate list. These variables
            // are collected while aggregating the inner loops. See
            // aggregateDef() for more details.
            Section.MAP may_set = may_map.get(loop);
            pri_set.addAll(may_set.keySet());
        } else {
            out = new Section.MAP();
            Object o = CFGraph.getIR(node);
            if (o instanceof Traversable) {
                Traversable tr = (Traversable)o;
                for (Expression e : DataFlowTools.getDefSet(tr)) {
                    out = out.unionWith(getSectionMap(e, true), rd);
                }
                killed_vars.addAll(DataFlowTools.getDefSymbol(tr));
                // Kill DEF section containing globals and actual parameters.
                in.removeSideAffected(tr);
            }
        }
        // Kill DEF section containing killed variables.
        in.removeAffected(killed_vars);
        // Candidate collection
        setPrivateCandidates(out);
        Section.MAP unioned = in.unionWith(out, rd);
        node.putData("def-out", unioned);
    }

    /**
    * Solves additive progressions for each node; this information is used when
    * aggregating MUST sections; the following equation describes this problem.
    *
    * IN[n]  = (V) { OUT[p], p in predecessors of n }
    * OUT[n] = IN[n] (^) GEN[n]
    * (v): join like constant propagation
    * (^): add induction increments
    *
    * Each variables' states are initialized to 0.
    * Non-induction assignment just kills the data.
    * Example with OUT data:
    * i=i+1 : i -&gt; 1
    * i=i+2 : i -&gt; 3
    * i=k   : i -&gt; TOP
    */
    private void computeLoopInduction(CFGraph g) {
        TreeMap<Integer, DFANode> work_list = new TreeMap<Integer, DFANode>();
        // Enter the entry node in the worklist.
        DFANode entry = g.getNodeWith("stmt", "ENTRY");
        work_list.put((Integer)entry.getData("top-order"), entry);
        boolean in_loop_body = false;
        // Do iterative steps
        while (!work_list.isEmpty()) {
            DFANode node = work_list.remove(work_list.firstKey());
            Map<Symbol, Expression> curr_map = null;
            for (DFANode pred:node.getPreds()) {
                Map<Symbol, Expression> pred_map = pred.getData("ind-out");
                // Unreachable nodes do not have "ind-out".
                if (pred_map == null) {
                    pred_map = new LinkedHashMap<Symbol, Expression>();
                }
                if (curr_map == null) {
                    curr_map = new LinkedHashMap<Symbol, Expression>(pred_map);
                } else {
                    curr_map = joinInductionMap(curr_map, pred_map);
                }
            }
            if (curr_map == null) {
                curr_map = new LinkedHashMap<Symbol, Expression>();
            }
            Map<Symbol, Expression> prev_map = node.getData("ind-in");
            if (prev_map == null || !prev_map.equals(curr_map)) {
                node.putData("ind-in", curr_map);
                if (in_loop_body) {
                    meetInductionMap(node);
                } else {
                    node.putData("ind-out", curr_map);
                    if (node.getData("header") != null) {
                        in_loop_body = true;
                    }
                }
                for (DFANode succ : node.getSuccs()) {
                    work_list.put((Integer)succ.getData("top-order"), succ);
                }
            }
        }
    }

    /**
    * Returns the result of a join operation of the two induction map.
    *  e1 v e2 = e1  if e1==e2,
    *          = T   otherwise 
    */
    private Map<Symbol, Expression> joinInductionMap(
            Map<Symbol, Expression> m1, Map<Symbol, Expression> m2) {
        Map<Symbol, Expression> ret = new LinkedHashMap<Symbol, Expression>();
        for (Symbol var : m1.keySet()) {
            Expression expr = m2.get(var);
            if (expr != null && expr.equals(m1.get(var))) {
                ret.put(var, expr);
            } else {
                ret.put(var, RangeExpression.getOmega());
            }
        }
        for (Symbol var : m2.keySet()) {
            ret.put(var, RangeExpression.getOmega());
        }
        return ret;
    }

    /**
    * Performs a meet operation in the specified node (transfer function).
    *  e1 ^ e2 = e1+e2  if e1!=T and e2!=T
    *          = T      otherwise
    */
    private void meetInductionMap(DFANode node) {
        Map<Symbol, Expression> in = node.getData("ind-in");
        Map<Symbol, Expression> out = new LinkedHashMap<Symbol, Expression>(in);
        Object o = CFGraph.getIR(node);
        if (!(o instanceof Traversable)) {
            node.putData("ind-out", out);
            return;
        }
        Traversable t = (Traversable)o;
        DFIterator<Traversable> iter = new DFIterator<Traversable>(t);
        Expression top = RangeExpression.getOmega();
        while (iter.hasNext()) {
            Traversable e = iter.next();
            Symbol var = null;
            Expression diff = null;
            if (e instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression)e;
                if (ae.getLHS() instanceof Identifier &&
                        SymbolTools.isInteger(
                        ((Identifier)ae.getLHS()).getSymbol())) {
                    var = ((Identifier)ae.getLHS()).getSymbol();
                    AssignmentOperator ao = ae.getOperator();
                    if (ao == AssignmentOperator.NORMAL) {
                        diff = Symbolic.subtract(ae.getRHS(), ae.getLHS());
                        if (IRTools.containsSymbol(diff, var)) {
                            diff = top;
                        }
                    } else if (ao == AssignmentOperator.ADD) {
                        diff = ae.getRHS().clone();
                    } else if (ao == AssignmentOperator.SUBTRACT) {
                        diff = Symbolic.subtract(new IntegerLiteral(0),
                                                 ae.getRHS());
                    } else {
                        diff = top;
                    }
                }
            } else if (e instanceof UnaryExpression) {
                UnaryExpression ue = (UnaryExpression)e;
                if (ue.getExpression() instanceof Identifier &&
                        SymbolTools.isInteger(
                        ((Identifier)ue.getExpression()).getSymbol())) {
                    var = ((Identifier)ue.getExpression()).getSymbol();
                    UnaryOperator uo = ue.getOperator();
                    if (uo.toString().equals("++")) {
                        diff = new IntegerLiteral(1);
                    } else if (uo.toString().equals("--")) {
                        diff = new IntegerLiteral(-1);
                    } else {
                        var = null;
                    }
                }
            }
            if (var != null) {
                if (IRTools.containsClass(t, ConditionalExpression.class)) {
                    diff = top; // Conservatively assume this is a may diff
                }
                Expression prev_diff = out.get(var);
                if (prev_diff == null) {
                    out.put(var, diff);
                } else if (prev_diff.equals(top) || diff.equals(top)) {
                    out.put(var, top);
                } else {
                    out.put(var, Symbolic.add(prev_diff, diff));
                }
            }
        }
        node.putData("ind-out", out);
    }

    /**
    * Examines each function call and collect arrays that may be used in the
    * function's callee. Scalars are not considered in this method because
    * whether they (scalars) are used is not important for the analysis;
    * remember that this is for conservative array section operations in the
    * absence of any interprocedural analysis.
    */
    private Set<Symbol> getMayUseFromFC(Traversable tr, Set<Symbol> vars) {
        Set<Symbol> ret = new LinkedHashSet<Symbol>();
        Set<Symbol> local_array = new LinkedHashSet<Symbol>();
        for (Symbol var : vars) {
            if (SymbolTools.isArray(var)) {
                if (SymbolTools.isGlobal(var, tr)) {
                    ret.add(var);
                } else {
                    local_array.add(var);
                }
            }
        }
        DFIterator<FunctionCall> iter =
                new DFIterator<FunctionCall>(tr, FunctionCall.class);
        iter.pruneOn(FunctionCall.class);
        while (iter.hasNext()) {
            FunctionCall fc = iter.next();
            for (Symbol var : local_array) {
                if (IRTools.containsSymbol(fc, var)) {
                    ret.add(var);
                }
            }
        }
        return ret;
    }

    /**
    * Collects candidate private variables from the section map by checking if
    * they do not contain any loop-variant variables.
    */
    private void setPrivateCandidates(Section.MAP m) {
        Set<Symbol> pri_set = pri_map.get(current_loop);
        for (Symbol var : m.keySet()) {
            Section section = m.get(var);
            if (section.isScalar()) {
                pri_set.add(var);
            } else if (!section.containsSymbols(loop_variants)) {
                pri_set.add(var);
            } else if (pri_set.contains(var)) {
                pri_set.remove(var);
            }
        }
    }

    /**
    * Computes the DEF summary set.
    */
    private Section.MAP getDefSet(CFGraph g) {
        Section.MAP ret = null;
        exit_range = null;
        exit_ind = null;
        // exit_range is a common range domain (union) for the exiting nodes.
        for (int i = 0; i < g.size(); ++i) {
            DFANode node = g.getNode(i);
            // Skip unreachable nodes (see HSG generation in CFGraph)
            Integer top_order = node.getData("top-order");
            if (top_order == null || top_order < 0) {
                continue;
            }
            if (node.getSuccs().size() > 0) {
                continue;
            }
            RangeDomain rd = node.getData("range");
            Map<Symbol, Expression> curr_ind = node.getData("ind-out");
            // Node with no successors is outside of the loop body, so let's
            // take def-in instead of def-out to avoid data kills due to the
            // loop step increments.
            Section.MAP curr_map = node.getData("def-in");
            if (ret == null) {
                exit_range = rd.clone();
                exit_ind = curr_ind;
                ret = curr_map.clone();
            } else {
                exit_range.unionRanges(rd);
                exit_ind = joinInductionMap(exit_ind, curr_ind);
                ret = ret.intersectWith(curr_map, exit_range);
            }
        }
        ret.clean();
        return ret;
    }

    /**
    * Computes the UEU summary set; UEU = UEU + (USE-DEF)
    */
    private Section.MAP getUseSet(CFGraph cfg) {
        Section.MAP ret = new Section.MAP();
        Iterator<DFANode> cfgiter = cfg.iterator();
        while (cfgiter.hasNext()) {
            DFANode node = cfgiter.next();
            // Skip unreachable nodes (see HSG generation in CFGraph)
            Integer top_order = node.getData("top-order");
            if (top_order == null || top_order < 0) {
                continue;
            }
            RangeDomain rd = node.getData("range");
            Section.MAP local_use = new Section.MAP();
            // Super node
            if (node.getData("super-entry") != null) {
                local_use = use_map.get(node.getData("super-entry"));
            // Other node
            } else {
                Object o = CFGraph.getIR(node);
                if (!(o instanceof Traversable)) {
                    continue;
                }
                for (Expression e : DataFlowTools.getUseSet((Traversable)o)) {
                    local_use = local_use.unionWith(
                            getSectionMap(e, false), rd);
                }
            }
            Section.MAP in_def = node.getData("def-in");
            //Section.MAP local_ueu = local_use.differenceFrom(in_def, rd);
            Section.MAP local_ueu = local_use.differenceFrom2(in_def, rd);
            ret = ret.unionWith(local_ueu, rd);
        }
        return ret;
    }

    /**
    * Collects privatizable variables. The candidate private variables have
    * already been stored in the pri_map for the analyzer. If the summary UEU
    * set of each loop contains any candidate variable, that variable is removed
    * from the candidate list. Another set of candidates come from the may_set
    * which contains the defined set of the inner loops with the execution of
    * the loop not guaranteed at compile time. This set needs to be added to the
    * private list after excluding UEU set.
    */
    private void collectPrivateSet(CFGraph g) {
        Set<Symbol> pri_set = pri_map.get(current_loop);
        Section.MAP may_set = new Section.MAP();
        // Do not collect for certain loops that are not parallelizable.
        // While loops, Do loops and loops containing conditional exits.
        // Other information such as DEF/USE set is still useful for those loops
        // for the analysis of outer loops.
        if (current_loop instanceof ForLoop) {
            Iterator<DFANode> node_iter = g.iterator();
            while (node_iter.hasNext()) {
                DFANode node = node_iter.next();
                Object ir = CFGraph.getIR(node);
                // 1. GotoStatement getting out of the loop.
                // 2. BreakStatement that breaks the current loop.
                // 3. ReturnStatement.
                if (ir instanceof GotoStatement &&
                    g.getNode("ir", ((GotoStatement)ir).getTarget()) == null ||
                    ir instanceof BreakStatement && node.getSuccs().isEmpty() ||
                    ir instanceof ReturnStatement) {
                    pri_set.clear();
                    continue;
                }
                // Collect delayed private variables.
                Statement loop = node.getData("super-entry");
                if (loop != null) {
                    Section.MAP loop_may_set = may_map.get(loop);
                    may_set = may_set.unionWith(loop_may_set, exit_range);
                }
            }
        } else {
            pri_set.clear();
        }
        Set<Symbol> use_set = use_map.get(current_loop).keySet();
        Set<Symbol> def_set = def_map.get(current_loop).keySet();
        // Include private variables in the inner loops, that are not used.
        DFIterator<Loop> iter =
                new DFIterator<Loop>(current_loop.getBody(), Loop.class);
        iter.pruneOn(Loop.class);
        iter.pruneOn(StatementExpression.class);
        while (iter.hasNext()) {
            Set<Symbol> inner_pri =
                    new LinkedHashSet<Symbol>(pri_map.get(iter.next()));
            inner_pri.removeAll(use_set);
            //inner_pri.removeAll(def_set);
            pri_set.addAll(inner_pri);
        }
        // Locally declared variables
        Set<Symbol> local_set = SymbolTools.getLocalSymbols(current_loop);
        pri_set.addAll(local_set);
        // Remove local_set from the DEF/USE set
        def_set.removeAll(local_set);
        use_set.removeAll(local_set);
        // Collect private variables
        pri_set.removeAll(use_set);
        // Variables that may be defined in inner loops need to be tested if
        // there is any read access to these variables. If there is no use
        // point for those variables, they are still considered to be private.
        may_set.keySet().removeAll(use_set);
        pri_set.addAll(may_set.keySet());
        Section.MAP recovered_def = def_map.get(current_loop);
        def_map.put(current_loop, recovered_def.unionWith(may_set, exit_range));
        // Conservatively remove variables from the private set if they appear
        // in function call parameters or they are global; will be improved.
        removeMayUsedVariables(pri_set);
        removeMayUsedVariables(def_set);
        // Remove any user-declared type; will be improved.
        removeUserTypes(pri_set);
        removeUserTypes(def_set);
        PrintTools.printlnStatus(2, tag, "DEF =", def_map.get(current_loop));
        PrintTools.printlnStatus(2, tag, "USE =", use_map.get(current_loop));
        PrintTools.printlnStatus(2, tag, "PRI =", pri_map.get(current_loop));
    }

    /**
    * Removes any variables typed with user-defined types.
    */
    private void removeUserTypes(Set<Symbol> vars) {
        Set<Symbol> keys = new LinkedHashSet<Symbol>(vars);
        for (Symbol var : keys) {
            if (var == null) {
                vars.remove(var);
                continue;
            }
            List types = var.getTypeSpecifiers();
            if (types != null) {
                for (Object type : types) {
                    if (type instanceof UserSpecifier) {
                        vars.remove(var);
                        break;
                    }
                }
            }
        }
    }

    /**
    * Removes variables from the set if they appear in function call or they
    * are global when a function call exists.
    */
    private void removeMayUsedVariables(Set<Symbol> vars) {
        boolean contains_unsafe_fcall = false;
        DFIterator<FunctionCall> iter =
                new DFIterator<FunctionCall>(current_loop, FunctionCall.class);
        while (iter.hasNext()) {
            FunctionCall fc = iter.next();
            if (StandardLibrary.isSideEffectFree(fc)) {
                continue;
            }
            contains_unsafe_fcall = true;
            // Referencing
            List<UnaryExpression> address_of =
                    IRTools.getUnaryExpression(fc, UnaryOperator.ADDRESS_OF);
            for (int i = 0; i < address_of.size(); i++) {
                UnaryExpression ue = address_of.get(i);
                vars.remove(SymbolTools.getSymbolOf(ue.getExpression()));
            }
            // Pointer type and user type
            Set<Symbol> params = SymbolTools.getAccessedSymbols(fc);
            //vars.removeAll(params) ; conservative
            for (Symbol param : params) {
                if (vars.contains(param)) {
                    List spec = param.getTypeSpecifiers();
                    if (Tools.containsClass(spec, PointerSpecifier.class) ||
                        Tools.containsClass(spec, UserSpecifier.class)) {
                        vars.remove(param);
                    }
                }
            }
        }
        // Global variables.
        if (contains_unsafe_fcall) {
            Iterator<Symbol> var_iter = vars.iterator();
            while (var_iter.hasNext()) {
                if (SymbolTools.isGlobal(var_iter.next(), current_loop)) {
                    var_iter.remove();
                }
            }
        }
    }

    /**
    * Computes set of live variables at each program point with the given
    * mask_set being the universal set.
    */
    private CFGraph getLiveVariables(Traversable t, Set<Symbol> mask_set) {
        CFGraph g = new CFGraph(t);
        g.topologicalSort(g.getNodeWith("stmt", "ENTRY"));
        TreeMap<Integer, DFANode> work_list = new TreeMap<Integer, DFANode>();
        List<DFANode> exit_nodes = g.getExitNodes();
        for (DFANode exit_node : exit_nodes) {
            work_list.put((Integer)exit_node.getData("top-order"), exit_node);
        }
        while (!work_list.isEmpty()) {
            DFANode node = work_list.remove(work_list.lastKey());
            // LIVEout
            Set<Symbol> live_out = new LinkedHashSet<Symbol>();
            for (DFANode succ : node.getSuccs()) {
                Set<Symbol> succ_in = succ.getData("live-in");
                if (succ_in != null) {
                    live_out.addAll(succ_in);
                }
            }
            // Convergence
            Set<Symbol> prev_live_out = node.getData("live-out");
            if (prev_live_out != null && prev_live_out.equals(live_out)) {
                continue;
            }
            node.putData("live-out", new LinkedHashSet<Symbol>(live_out));
            // Local computation
            Set<Symbol> gen = new LinkedHashSet<Symbol>();
            Set<Symbol> kill = new LinkedHashSet<Symbol>();
            computeLiveVariables(node, gen, kill, mask_set);
            // LiveIn = Gen (v) ( LiveOut - Kill )
            live_out.removeAll(kill);
            live_out.addAll(gen);
            // Intersect with the masking set (reduces the size of the live set)
            live_out.retainAll(mask_set);
            node.putData("live-in", live_out);
            for (DFANode pred : node.getPreds()) {
                work_list.put((Integer)pred.getData("top-order"), pred);
            }
        }
        return g;
    }

    /**
    * Transfer function for live variable analysis.
    */
    private void computeLiveVariables(DFANode node,
                                      Set<Symbol> gen,
                                      Set<Symbol> kill,
                                      Set<Symbol> mask_set) {
        Object tr = CFGraph.getIR(node);
        if (!(tr instanceof Traversable)) { // symbol-entry with initializer?
            return;
        }
        gen.addAll(DataFlowTools.getUseSymbol((Traversable)tr));
        kill.addAll(DataFlowTools.getDefSymbol((Traversable)tr));
        // Conservative decision on funcion calls; add any variables in the
        // mask_set to the GEN set.
        if (IRTools.containsClass((Traversable)tr, FunctionCall.class)) {
            for (Symbol var : mask_set) {
                if (SymbolTools.isGlobal(var, current_loop)) {
                    gen.add(var);
                }
            }
        }
        // Name only support for access expressions.
        addAccessName(gen);
        addAccessName(kill);
        return;
    }

    /**
    * Adds the base name of an access expression to the given set to enable
    * name-only analysis.
    */
    private void addAccessName(Set<Symbol> set) {
        Set<Symbol> vars = new LinkedHashSet<Symbol>(set);
        for (Symbol var : vars) {
            if (var instanceof AccessSymbol) {
                set.add(((AccessSymbol)var).getIRSymbol());
            }
        }
    }

    /**
    * Aggregates MUST DEF set of the loop.
    */
    private void aggregateDef() {
        // Additional information about zero-trip loop consideration.
        RangeDomain entry_range = RangeAnalysis.query((Statement)current_loop);
        if (entry_range == null) {
            entry_range = new RangeDomain();
        } else {
            entry_range = entry_range.clone();
        }
        int eval = 1;   // eval == 1 means the loop is executed at least once.
        Set<Symbol> init_set = new LinkedHashSet<Symbol>();
        if (current_loop instanceof ForLoop) {
            ForLoop for_loop = (ForLoop)current_loop;
            Statement i_stmt = for_loop.getInitialStatement();
            // Add range from the initial statement.
            entry_range.intersectRanges(RangeAnalysis.getRangeDomain(i_stmt));
            eval = entry_range.evaluateLogic(for_loop.getCondition());
            init_set.addAll(DataFlowTools.getDefSymbol(i_stmt));
        } else if (current_loop instanceof WhileLoop) {
            WhileLoop while_loop = (WhileLoop)current_loop;
            eval = entry_range.evaluateLogic(while_loop.getCondition());
        }
        Section.MAP may_set = new Section.MAP();
        Section.MAP defs = def_map.get(current_loop);
        Set<Symbol> vars = new LinkedHashSet<Symbol>(defs.keySet());
        for (Symbol var : vars) {
            Set<Symbol> ivs = getInductionVariable();
            Section before = defs.get(var);
            Section section = before.clone();
            section.expandMust(exit_range, ivs, loop_variants);
            section.substituteForward(exit_range, loop_variants);
            if (section.isArray() && section.isEmpty()) {
                defs.remove(var);
                continue;
            }
            // If the loop is not executed in any case, move the sections to
            // the may_set and the outer loop makes decision whether the
            // elements of the may_set is private.
            if (section.equals(before)) {
                if (eval != 1 && !init_set.contains(var)) {
                    may_set.put(var, section);
                    defs.remove(var);
                }
            } else {
                defs.put(var, section);
            }
        }
        may_map.put(current_loop, may_set);
        PrintTools.printlnStatus(2, tag, "DEFS =", defs, "under", exit_range);
    }

    /**
    * Aggregates MAY USE set of the loop.
    */
    private void aggregateUse() {
        Section.MAP my_use = use_map.get(current_loop);
        Set<Symbol> vars = new LinkedHashSet<Symbol>(my_use.keySet());
        for (Symbol var : vars) {
            Section section = my_use.get(var);
            section.expandMay(exit_range, loop_variants);
            section.substituteForward(exit_range, loop_variants);
        }
        PrintTools.printlnStatus(2, tag, "USES =", my_use, "under", exit_range);
    }

    /**
    * Returns a set of induction variables which are used in the aggregation
    * process in the future.
    */
    private Set<Symbol> getInductionVariable() {
        Set<Symbol> ret = new LinkedHashSet<Symbol>();
        for (Symbol var : exit_ind.keySet()) {
            Expression stride = exit_ind.get(var);
            if (stride.equals(new IntegerLiteral(1))) {
                ret.add(var);
            }
        }
        return ret;
    }

    /**
    * Returns map from a variable to its section
    */
    private Section.MAP getSectionMap(Expression e, boolean def) {
        Section.MAP ret = new Section.MAP();
        if (e instanceof ArrayAccess) {
            ArrayAccess aa = (ArrayAccess)e;
            Symbol var = SymbolTools.getSymbolOf(aa.getArrayName());
            if (var instanceof AccessSymbol) {
                // Only use set is considered important for name-only analysis
                // because it implies more conservative analysis.
                if (!def) {
                    ret.put(((AccessSymbol)var).getIRSymbol(), new Section(-1));
                }
            } else {
                ret.put(SymbolTools.getSymbolOf(aa.getArrayName()),
                        new Section(aa));
            }
        } else if (e instanceof AccessExpression) {
            if (!def) {
                Set use_set = DataFlowTools.getUseSet(e);
                if (use_set.size() == 1) {
                    AccessSymbol var = (AccessSymbol)SymbolTools.getSymbolOf(e);
                    ret.put(var.getIRSymbol(), new Section(-1));
                }
            }
        } else {
            Symbol var = SymbolTools.getSymbolOf(e);
            // var == null means it is not variable type
            // e.g.) *a = 0;
            if (var != null) {
                ret.put(var, new Section(-1));
            }
        }
        ret.clean();
        return ret;
    }

    /**
    * For now, just union them once in reverse post order.
    * This method is inefficient because we need to rebuild RangeDomain for
    * intermediate nodes which do not represent a statement in the IR.
    * It seems like this is the only way to provide range information correctly.
    */
    private void getRangeDomain(CFGraph g) {
        TreeMap<Integer, DFANode> work_list = new TreeMap<Integer, DFANode>();
        Iterator<DFANode> iter = g.iterator();
        while (iter.hasNext()) {
            DFANode node = iter.next();
            work_list.put((Integer)node.getData("top-order"), node);
        }
        for (Integer order : work_list.keySet()) {
            DFANode node = work_list.get(order);
            Object ir = node.getData(Arrays.asList("super-entry", "stmt"));
            RangeDomain rd = null;
            if (ir instanceof Statement) {
                rd = RangeAnalysis.query((Statement)ir);
            }
            if (rd != null) {
                node.putData("range", rd.clone());
            } else if (order == 0) {
                RangeDomain range =
                        RangeAnalysis.query((Statement)current_loop);
                if (range == null) {
                    node.putData("range", new RangeDomain());
                } else {
                    node.putData("range", range.clone());
                }
            } else {
                RangeDomain range = null;
                for (DFANode pred : node.getPreds()) {
                    RangeDomain pred_range = pred.getData("range");
                    if (pred_range == null) {
                        pred_range = new RangeDomain();
                    }
                    if (range == null) {
                        range = pred_range.clone();
                    } else {
                        range.unionRanges(pred_range);
                    }
                }
                node.putData("range", range);
            }
        }
    }

}
