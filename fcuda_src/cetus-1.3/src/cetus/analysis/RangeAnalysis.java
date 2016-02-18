package cetus.analysis;

import cetus.exec.Driver;
import cetus.hir.*;

import java.util.*;

/**
 * RangeAnalysis performs symbolic range propagation for the programs by
 * symbolically executing the program and abstracting the values of integer
 * variables with a symbolic bounds. The implementation is based on Symbolic
 * Range Propagation by Blume and Eigenmann, which was implemented for the
 * Fortran77 language.
 * <p>
 * The goal of Range Analysis is to collect, at each program statement, a map
 * from integer-typed scalar variables to their symbolic value ranges,
 * represented by a symbolic lower bound and an upper bound.
 * In other words, a symbolic value range expresses the relationship between the
 * variables that appear in the range. We use a similar approach as in Symbolic
 * Range Propagation in the Polaris parallelizing compiler, with necessary
 * adjustment for the C language, to compute the set of value ranges before
 * each statement. The set of value ranges at each statement can be used in
 * several ways. Pass writers can directly query the symbolic bound of a
 * variable or can compare two symbolic expressions using the constraints given
 * by the set of value ranges.
 * <p>
 * The high-level algorithm performs fix-point iteration in two phases when
 * propagating the value ranges throughout the program.
 * The first phase applies widening operations at nodes that have incoming back
 * edges to guarantee the termination of the algorithm. The second phase
 * compensates the loss of information due to the widening operations by
 * applying narrowing operation to the node at which widening has occurred.
 * During the fix-point iteration, the value ranges are merged at nodes that
 * have multiple predecessors, and outgoing value ranges are computed by
 * symbolically executing the statement. Two typical types of program semantics
 * that cause such changes of value ranges are constraints from conditional
 * expressions and assignments to variables.
 * <p>
 * Range analysis returns a map from each statement to its corresponding
 * {@link RangeDomain} that contains the set of valid value ranges before
 * each statement. The result of this analysis was verified with the
 * C benchmarks in the SPEC CPU2006 suite. Range analysis does not handle
 * integer overflow as it does not model the overflow but such cases are rare
 * in normal correct programs. Following example shows how to invoke a range
 * analyzer in a compiler pass:
 * <pre>
 * Map&lt;Statement, RangeDomain&gt; range_map = RangeAnalysis.getRanges(procedure);
 * RangeDomain range_domain = range_map.get(statement);
 * // range_domain now contains the set of value ranges for the statement.
 * </pre>
 * Following example shows a function and its range map created after the
 * range analysis.
 * <pre>
 *          int foo(int k) {                             
 *                  []                    
 *            int i, j;                   
 *                  []                    
 *            double a;                   
 *                  []                    
 *            for ( i=0; i&lt;10; ++i ) {                           
 *                  [0&lt;=i&lt;=9]             
 *              a=(0.5*i);                
 *            }                           
 *                  [i=10]                
 *            j=(i+k);                    
 *                  [i=10, j=(i+k)]       
 *            return j;                   
 *          }                             
 * </pre>
 */
public class RangeAnalysis extends AnalysisPass
{
    /** Pass tag */
    private static final String tag = "[RangeAnalysis]";
    /** Set of tractable expression types */
    private static final Set<Class<? extends Expression>> tractable_class;
    /** Set of tractable operation types */
    private static final Set<Printable> tractable_op;
    /** Debug level */
    private static int debug;

    /** Option for disabling range computation */
    public static final int RANGE_EMPTY = 0;
    /** Option for enforcing intra-procedural analysis */
    public static final int RANGE_INTRA = 1;
    /** Option for enforcing inter-procedural analysis */
    public static final int RANGE_INTER = 2;
    /** Option for enforcing use of range pragma and constraint */
    public static final int RANGE_PRAGMA = 3; // TODO: publishable?
    /** Range analysis option */
    private static int option;
    /** Read-only Literals */
    private static final IntegerLiteral one = new IntegerLiteral(1);

    static {
        tractable_class = new HashSet<Class<? extends Expression>>();
        tractable_class.add(ArrayAccess.class);
        tractable_class.add(BinaryExpression.class);
        tractable_class.add(Identifier.class);
        tractable_class.add(InfExpression.class);
        tractable_class.add(IntegerLiteral.class);
        tractable_class.add(MinMaxExpression.class);
        tractable_class.add(RangeExpression.class);
        tractable_class.add(UnaryExpression.class);
        tractable_op = new HashSet<Printable>();
        tractable_op.add(BinaryOperator.ADD);
        tractable_op.add(BinaryOperator.DIVIDE);
        tractable_op.add(BinaryOperator.MODULUS);
        tractable_op.add(BinaryOperator.MULTIPLY);
        tractable_op.add(UnaryOperator.MINUS);
        tractable_op.add(UnaryOperator.PLUS);
        debug = PrintTools.getVerbosity();
        option = RANGE_INTRA;
        if (Driver.getOptionValue("range") != null) {
            option = Integer.parseInt(Driver.getOptionValue("range"));
        }
    }

    /**
    * Interprocedural input.
    * ip_node: the procedure node to be processed.
    * ip_cfg : the procedure control flow graph to be processed.
    */
    private static IPANode ip_node = null;
    private static CFGraph ip_cfg = null;

    /** Result of interprocedural range analysis. */
    private static Map<Procedure, Map<Statement, RangeDomain>> ip_ranges = null;

    /** Alias analysis for less conservative range analysis. */
    private static AliasAnalysis alias;

    /**
    * Past result of range analysis kept in a single map which needs
    * invalidation if a transformation pass is called
    */
    private static final Map<Statement, RangeDomain> range_domains =
            new HashMap<Statement, RangeDomain>();
    private static final RangeDomain empty_range = new RangeDomain();

    // TODO: experimental code
    private static final Set<String> safe_functions = new HashSet<String>();
    static {
        String s = System.getenv("CETUS_RANGE_SAFE_FUNCTIONS");
        if (s != null) {
            for (String safe_function : s.split(",")) {
                safe_functions.add(safe_function);
            }
        }
    }

    /**
    * Constructs a range analyzer for the program. The instantiated constructor
    * is used only internally.
    * @param program  the input program.
    */
    public RangeAnalysis(Program program) {
        super(program);
    }

    /**
    * Returns the pass name.
    * @return the pass name in string.
    */
    @Override
    public String getPassName() {
        return tag;
    }

    /**
    * Starts range analysis for the whole program. this is useful only for
    * testing the range analysis.
    */
    @Override
    public void start() {
        double timer = Tools.getTime();
        DFIterator<Procedure> iter =
                new DFIterator<Procedure>(program, Procedure.class);
        iter.pruneOn(Procedure.class);
        iter.pruneOn(Declaration.class);
        while (iter.hasNext()) {
            Procedure procedure = iter.next();
            Map<Statement, RangeDomain> ranges = getRanges(procedure);
            addAssertions(procedure, ranges);
            // Test codes for tools with range information.
            //testSubstituteForward((Procedure)o, range_map);
        }
        DFIterator<TranslationUnit> titer =
                new DFIterator<TranslationUnit>(program, TranslationUnit.class);
        titer.pruneOn(TranslationUnit.class);
        while (titer.hasNext()) {
            TranslationUnit tu = titer.next();
            Declaration ref = (Declaration)tu.getChildren().get(0);
            tu.addDeclarationBefore(ref, new AnnotationDeclaration(
                    new CodeAnnotation("#include <assert.h>")));
        }
        timer = Tools.getTime(timer);
        PrintTools.printlnStatus(1, tag, String.format("%.2f\n", timer));
    }

    // Adds assert() call for range validation.
    private void addAssertions(
            Procedure procedure, Map<Statement, RangeDomain> ranges) {
        DFIterator<Statement> iter =
                new DFIterator<Statement>(procedure, Statement.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            if (stmt instanceof DeclarationStatement ||
                stmt instanceof CompoundStatement &&
                    !(stmt.getParent() instanceof CompoundStatement)) {
                continue;
            } // Skip program point that doesn't have a state.
            RangeDomain rd = ranges.get(stmt);
            if (rd != null && rd.size() > 0) {
                Expression test = rd.toExpression();
                stmt.annotateBefore(
                        new CodeAnnotation("assert(" + test + ");"));
            }
        }
    }

    // Test routine for substituteForward
    private void testSubstituteForward(
            Procedure proc, Map<Statement, RangeDomain> range_map) {
        DFIterator<Statement> iter =
                new DFIterator<Statement>(proc, Statement.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            RangeDomain rd = range_map.get(stmt);
            if (rd == null)
                continue;
            PrintTools.printlnStatus(0, "[RD]", "=", rd);
            Set<Expression> use = DataFlowTools.getUseSet(stmt);
            for (Expression e : use) {
                Expression substituted = rd.substituteForward(e);
                PrintTools.printlnStatus(0,
                        tag, "[SubstituteForward]", e, "=>", substituted);
            }
        }
    }

    /**
    * Performs range analysis for the procedure and returns the result.
    * @param proc the input procedure.
    * @return the range map for the procedure.
    */
    public Map<Statement, RangeDomain> getRangeMap(Procedure proc) {
        return getRanges(proc);
    }

    /**
    * Generates a range map from the specified control flow graph and the root
    * traversable object.
    * @param root the requested root node of the IR tree.
    * @param cfg the control flow graph decorated with value ranges.
    * @return the map from statements to their range domain.
    */
    private static Map<Statement, RangeDomain>
            getRangeMap(Traversable root, CFGraph cfg) {
        // Step 1: Filters nodes without any associated statement.
        Map<Statement, RangeDomain> ret = new HashMap<Statement, RangeDomain>();
        for (int i = 0; i < cfg.size(); i++) {
            DFANode node = cfg.getNode(i);
            Object o = null;
            RangeDomain rd = null;
            if ((o=node.getData("stmt")) instanceof Statement &&
                (rd=node.getData("ranges")) != null) {
                ret.put((Statement)o, rd);
            }
        }
        // Step 2: Filters potentially unsafe range information. This is
        // required because CFGraph and Traversable cannot exchangeable each
        // other 100%, in terms of scope information.
        filterUnsafeRanges(root, ret);
        return ret;
    }

    /** 
    * Removes any range information that is not within a valid scope or does
    * contain pointer types (which is not closed under algebraic operations).
    * @param t the root object for the range information.
    * @param ranges the computed ranges for the root object.
    */
    private static void filterUnsafeRanges(
            Traversable t, Map<Statement, RangeDomain> ranges) {
        Map<SymbolTable, Set<Symbol>> scopes =
                new HashMap<SymbolTable, Set<Symbol>>();
        DFIterator<SymbolTable> iter =
                new DFIterator<SymbolTable>(t, SymbolTable.class);
        while (iter.hasNext()) {
            SymbolTable symtab = iter.next();
            scopes.put(symtab, symtab.getSymbols());
        }
        SymbolTable outer = IRTools.getAncestorOfType(t, SymbolTable.class);
        while (outer != null) {
            scopes.put(outer, outer.getSymbols());
            outer = IRTools.getAncestorOfType(outer, SymbolTable.class);
        }
        for (Statement stmt : ranges.keySet()) {
            RangeDomain rd = ranges.get(stmt);
            Iterator<Symbol> var_iter = rd.getSymbols().iterator();
            while (var_iter.hasNext()) {
                Symbol var = var_iter.next();
                if (isPointer(var) || !isWithinScope(stmt, var, scopes)) {
                    var_iter.remove();
                } else {
                    DFIterator<Identifier> id_iter = new DFIterator<Identifier>(
                            rd.getRange(var), Identifier.class);
                    while (id_iter.hasNext()) {
                        Symbol id_symbol = id_iter.next().getSymbol();
                        if (isPointer(id_symbol) ||
                            !isWithinScope(stmt, id_symbol, scopes)) {
                            var_iter.remove();
                            break;
                        }
                    }
                }
            }
        }
    }

    /** Checks if the given symbol is within a valid scope. */
    private static boolean isWithinScope(Traversable t, Symbol symbol,
            Map<SymbolTable, Set<Symbol>> scopes) {
        SymbolTable symtab = IRTools.getAncestorOfType(t, SymbolTable.class);
        while (symtab != null) {
            Set<Symbol> scope = scopes.get(symtab);
            if (scope != null && scope.contains(symbol)) {
                return true;
            } else {
                symtab = IRTools.getAncestorOfType(symtab, SymbolTable.class);
            }
        }
        return false;
    }

    /** Optimized check for pointer types */
    private static boolean isPointer(Symbol symbol) {
        return (
        SymbolTools.containsSpecifier(symbol, PointerSpecifier.UNQUALIFIED) ||
        SymbolTools.containsSpecifier(symbol, PointerSpecifier.CONST) ||
        SymbolTools.containsSpecifier(symbol, PointerSpecifier.VOLATILE) ||
        SymbolTools.containsSpecifier(symbol, PointerSpecifier.CONST_VOLATILE));
    }

    /** Optimized check for integer types */
    private static boolean isInteger(Symbol symbol) {
        return (
        !isPointer(symbol) &&
        !SymbolTools.containsSpecifier(symbol, Specifier.CHAR) && (
        SymbolTools.containsSpecifier(symbol, Specifier.INT) ||
        SymbolTools.containsSpecifier(symbol, Specifier.LONG) ||
        SymbolTools.containsSpecifier(symbol, Specifier.SIGNED) ||
        SymbolTools.containsSpecifier(symbol, Specifier.UNSIGNED)));
    }

    /**
    * Returns the range map in a pretty format.
    * @param t        the traversable object.
    * @param ranges   the range map.
    * @param indent   the indent for pretty printing.
    * @return         the string in a pretty format.
    */
    public static String toPrettyRanges(
            Traversable t, Map<Statement, RangeDomain> ranges, Integer indent) {
        StringBuilder sb = new StringBuilder(80);
        if (t instanceof Statement) {
            RangeDomain rd = ranges.get(t);
            sb.append("        ");
            if (rd == null) {
                sb.append("[]");
                //sb.append(" size=0");
            } else {
                sb.append(rd);
                //sb.append(" size=");
                //sb.append(rd.size());
            }
            sb.append(PrintTools.line_sep);
        }
        String tab = "";
        for (int i = 0; i < indent; ++i) {
            tab += "  ";
        }
        if (t instanceof Procedure) {
            Procedure p = (Procedure)t;
            sb.append(tag);
            sb.append(" Range Domain for Procedure ");
            sb.append(p.getName());
            sb.append(PrintTools.line_sep);
            sb.append(toPrettyRanges(p.getBody(), ranges, indent));
        } else if (t instanceof CompoundStatement) {
            sb.append(tab);
            sb.append("{");
            sb.append(PrintTools.line_sep);
            indent++;
            for (Traversable child : t.getChildren()) {
                sb.append(toPrettyRanges(child, ranges, indent));
            }
            indent--;
            sb.append(tab);
            sb.append("}");
            sb.append(PrintTools.line_sep);
        } else if (t instanceof DoLoop) {
            DoLoop d = (DoLoop)t;
            sb.append(tab);
            sb.append("do");
            sb.append(PrintTools.line_sep);
            sb.append(toPrettyRanges(d.getBody(), ranges, indent));
            sb.append(tab);
            sb.append("while (");
            sb.append(d.getCondition());
            sb.append(");");
            sb.append(PrintTools.line_sep);
        } else if (t instanceof ForLoop) {
            ForLoop f = (ForLoop)t;
            sb.append(tab);
            sb.append("for (");
            Traversable child = f.getInitialStatement();
            if (child == null) {
                sb.append(";");
            } else {
                sb.append(child);
            }
            sb.append(" ");
            child = f.getCondition();
            if (child == null) {
                sb.append(" ");
            } else {
                sb.append(child);
            }
            sb.append("; ");
            child = f.getStep();
            if (child != null) {
                sb.append(child);
            }
            sb.append(")");
            sb.append(PrintTools.line_sep);
            sb.append(toPrettyRanges(f.getBody(), ranges, indent));
        } else if (t instanceof IfStatement) {
            IfStatement i = (IfStatement)t;
            sb.append(tab);
            sb.append("if (");
            sb.append(i.getControlExpression());
            sb.append(")");
            sb.append(PrintTools.line_sep);
            sb.append(toPrettyRanges(i.getThenStatement(), ranges, indent));
            Statement els = i.getElseStatement();
            if (els != null) {
                sb.append("else");
                sb.append(PrintTools.line_sep);
                sb.append(toPrettyRanges(els, ranges, indent));
            }
        } else if (t instanceof SwitchStatement) {
            SwitchStatement s = (SwitchStatement)t;
            sb.append(tab);
            sb.append("switch (");
            sb.append(s.getExpression());
            sb.append(")");
            sb.append(PrintTools.line_sep);
            sb.append(toPrettyRanges(s.getBody(), ranges, indent));
        } else if (t instanceof WhileLoop) {
            WhileLoop w = (WhileLoop)t;
            sb.append(tab);
            sb.append("while (");
            sb.append(w.getCondition());
            sb.append(")");
            sb.append(PrintTools.line_sep);
            sb.append(toPrettyRanges(w.getBody(), ranges, indent));
        } else if (t instanceof Statement) {
            sb.append(tab);
            sb.append(t);
            sb.append(PrintTools.line_sep);
        }
        return sb.toString();
    }

    // Returns an empty map of ranges - fallback behavior.
    private static Map<Statement, RangeDomain> getEmptyRanges(Traversable t) {
        Map<Statement, RangeDomain> ret = new HashMap<Statement, RangeDomain>();
        RangeDomain empty = new RangeDomain();
        DFIterator<Statement> iter =
                new DFIterator<Statement>(t, Statement.class);
        while (iter.hasNext()) {
            ret.put(iter.next(), empty); 
        }
        return ret;
    }

    /** Enforces use of the specified option for computing the ranges */
    public static Map<Statement, RangeDomain>
            getRanges(SymbolTable symtab, int temporal_option) {
        int original_option = option;
        option = temporal_option;
        Map<Statement, RangeDomain> ret = getRanges(symtab);
        option = original_option;
        return ret;
    }

    /**
    * Returns a range map created for the symbol table object. It is more
    * general to support range map for a traversable object but supporting this
    * functionality on an object that contains collection of statement object,
    * i.e. symbol table, is more natural.
    * @param symtab the symbol table object whose range map is collected.
    * @return the range map for the traversable object.
    */
    public static Map<Statement, RangeDomain> getRanges(SymbolTable symtab) {
        // Invokes alias analysis for less conservative analysis.
        alias = new AliasAnalysis(
                IRTools.getAncestorOfType(symtab, Program.class));
        alias.start();

        Map<Statement, RangeDomain> ret = null;
        if (symtab instanceof TranslationUnit) {
            PrintTools.printlnStatus(0,
                    tag, "Range map for the whole program is unavailable");
            return ret;
        } else if (symtab instanceof Procedure) {
            String proc_name = " " + ((Procedure)symtab).getSymbolName();
            switch (option) {
            case RANGE_INTER:
                if (ip_ranges == null) {
                    Program prog =
                            IRTools.getAncestorOfType(symtab, Program.class);
                    if (prog != null) {
                        PrintTools.printlnStatus(1,
                                tag, "Recomputing IP ranges......");
                        IPRangeAnalysis.compute(prog);
                    }
                }
                if (ip_ranges != null &&
                    (ret = ip_ranges.get(symtab)) != null) {
                    PrintTools.printlnStatus(1,
                            tag, proc_name, ": Retrieving IP ranges......");
                } else {
                    PrintTools.printlnStatus(1, tag, proc_name,
                            ": Falling back to intra analysis......");
                }
                break;
            case RANGE_EMPTY:
                PrintTools.printlnStatus(1, tag, proc_name,
                        ": Returning empty ranges as requested......");
                ret = getEmptyRanges(symtab);
                break;
            case RANGE_PRAGMA: // only intra analysis is possible for this mode
            case RANGE_INTRA:
            default:
                PrintTools.printlnStatus(1, tag, proc_name,
                        ": Computing local ranges......");
            }
            if (ret != null) {
                if (debug >= 2) {
                    System.err.println(
                            toPrettyRanges(symtab, ret, new Integer(0)));
                }
                return ret;
            }
        }

        Traversable root = symtab;
        while (root != null && root.getClass() != Program.class)
            root = root.getParent();

        CFGraph cfg = new CFGraph(symtab);
        cfg.normalize();
        cfg.topologicalSort(cfg.getNodeWith("stmt", "ENTRY"));
        iterateToFixpoint(cfg, true);
        iterateToFixpoint(cfg, false);
        if (debug >= 3) {
            System.err.println(cfg.toDot("top-order,ranges,ir,tag", 3));
        }

        ret = getRangeMap(symtab, cfg);

        // Inserts an empty range if there is no associated range domain.
        RangeDomain empty = new RangeDomain();
        DFIterator<Statement> iter =
                new DFIterator<Statement>(symtab, Statement.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            if (ret.get(stmt) == null) {
                ret.put(stmt, empty);
            }
        }
        if (debug >= 2) {
            System.err.println(toPrettyRanges(symtab, ret, new Integer(0)));
        }
        return ret;
    }

    /**
    * Returns the fine-grain result of range analysis with the given
    * interprocedural input. "node" represents a procedure whose IPA results are
    * stored inside. "ip_node" and "ip_cfg" are temporarily stored for a single
    * invocation of getRangeCFG, and are accessed during intraprocedural
    * analysis that interprets the flow entry of a procedure and the call site
    * inside a procedure.
    */
    protected static CFGraph getRangeCFG(IPANode node) {
        ip_node = node;
        debug = PrintTools.getVerbosity();
        CFGraph ret = new CFGraph(node.getProcedure());
        ret.normalize();
        DFANode entry_node = ret.getEntry();
        ret.topologicalSort(entry_node);
        ip_cfg = ret;
        Domain in0 = node.in();
        if (in0 instanceof RangeDomain)
            entry_node.putData("ranges", ((RangeDomain)in0).clone());
        iterateToFixpoint(ret, true);
        iterateToFixpoint(ret, false);
        ip_node = null;
        ip_cfg = null;
        return ret;
    }

    /**
    * Returns the coarse-grain result of range analysis with the given
    * interprocedural input. It internally calls getRangeCFG and jsut converts
    * the cfg-based workspace into a map from statements to range domains.
    */
    protected static Map<Statement, RangeDomain> getRanges(IPANode node) {
        Procedure proc = node.getProcedure();
        Map<Statement, RangeDomain> ret = getRangeMap(proc, getRangeCFG(node));
        RangeDomain empty = new RangeDomain();
        DFIterator<Statement> iter =
                new DFIterator<Statement>(proc, Statement.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            if (ret.get(stmt) == null) {
                ret.put(stmt, empty);
            }
        }
        if (debug >= 2) {
            System.err.println(toPrettyRanges(proc, ret, new Integer(0)));
        }
        return ret;
    }

    /**
    * Returns a range domain for the given statement.
    * @param stmt the input statement.
    * @return the out range domain for the statement.
    */
    public static RangeDomain getRangeDomain(Statement stmt) {
        if (stmt == null || !(stmt instanceof ExpressionStatement))
            return new RangeDomain();
        CFGraph cfg = new CFGraph(stmt);
        cfg.normalize();
        cfg.topologicalSort(cfg.getNodeWith("stmt", "ENTRY"));
        DFANode last = cfg.getNodeWith("top-order", new Integer(cfg.size()-1));
        DFANode exit = new DFANode("stmt", "EXIT");
        exit.putData("top-order", new Integer(cfg.size()));
        cfg.addEdge(last, exit);
        // Exit node is necessary because Range Analysis computes only in-range.
        // Maybe CFGraph needs flow exit always but it breaks the range
        // analysis, so it needs to be revisited later.
        iterateToFixpoint(cfg, true);
        RangeDomain ret = exit.getData("ranges");
        if (ret == null)
            ret = new RangeDomain();
        return ret;
    }

    /**
    * Sets the interprocedural range maps with the specified new map.
    */
    protected static void
            setRanges(Map<Procedure, Map<Statement, RangeDomain>> ip_ranges) {
        RangeAnalysis.ip_ranges = ip_ranges;
    }

    // Eliminates variables with pointer type.
    private static boolean containsPointerType(Traversable t) {
        if (t == null)
            return false;
        DFIterator<Identifier> iter =
                new DFIterator<Identifier>(t, Identifier.class);
        while (iter.hasNext()) {
            if (isPointer(iter.next().getSymbol())) {
                return true;
            }
        }
        return false;
    }

    // Marks the node having a back edge by comparing topological order
    private static void setBackedge(DFANode node) {
        int my_order = (Integer)node.getData("top-order");
        for (DFANode pred : node.getPreds()) {
            if (my_order < (Integer)pred.getData("top-order")) {
                node.putData("has-backedge", new Boolean(true));
                break;
            }
        }
    }

    // Fixpoint iteration
    private static void iterateToFixpoint(CFGraph g, boolean widen) {
        TreeMap<Integer, DFANode> work_list = new TreeMap<Integer, DFANode> ();
        // Add the entry node to the work list for widening phase.
        if (widen) {
            DFANode entry = g.getNodeWith("stmt", "ENTRY");
            if (entry.getData("ranges") == null)
                entry.putData("ranges", new RangeDomain());
            work_list.put((Integer) entry.getData("top-order"), entry);
        } else {
        // Add the widened nodes to the work list for narrowing phase.
            for (int i = 0; i < g.size(); ++i) {
                DFANode widened = g.getNode(i);
                if (widened.getData("has-backedge") != null) {
                    work_list.put((Integer)widened.getData("top-order"),
                                  widened);
                }
            }
        }
        while (work_list.size() > 0) {
            // Get the first node in topological order from the work list.
            Integer node_num = work_list.firstKey();
            DFANode node = work_list.remove(node_num);
            // Record number of iterations for each node.
            Integer visits = node.getData("num-visits");
            if (visits == null) {
                node.putData("num-visits", new Integer(1));
                setBackedge(node);
            } else {
                node.putData("num-visits", new Integer(visits + 1));
            }
            PrintTools.printlnStatus(3, tag, "Visited Node#", node_num);
            PrintTools.printlnStatus(3, tag, "  IR =",CFGraph.getIR(node));
            // Merge incoming states from predecessors.
            RangeDomain curr_ranges = null;
            for (DFANode pred : node.getPreds()) {
                RangeDomain pred_range_out = node.getPredData(pred);
                // Skip BOT-state predecessors that has not been visited.
                if (pred_range_out == null) {
                    continue;
                }
                if (curr_ranges == null) {
                    curr_ranges = new RangeDomain(pred_range_out);
                } else {
                    curr_ranges.unionRanges(pred_range_out);
                }
            }
            PrintTools.printlnStatus(3, tag, "  UNION =", curr_ranges);
            // Add initial values from declarations
            enterScope(node, curr_ranges);
            // Widening/Narrowing operations.
            RangeDomain prev_ranges = node.getData("ranges");
            if (prev_ranges != null && node.getData("has-backedge") != null) {
                if (widen) {
                    Set<Symbol> widener = node.getData("loop-variants");
                    // Selective widening only with loop-variant symbols.
                    if (widener != null && widener.size() > 0) {
                        curr_ranges.widenAffectedRanges(prev_ranges, widener);
                    } else {
                        curr_ranges.widenRanges(prev_ranges);
                    }
                } else {
                    curr_ranges.narrowRanges(prev_ranges);
                }
            }
            PrintTools.printlnStatus(3, tag, "  WIDEN/NARROW =", curr_ranges);
            if (option == RANGE_PRAGMA) {
                // Reads user-provided range pragma.
                if (curr_ranges != null) {
                    curr_ranges.intersectRanges(extractRangePragma(node));
                }
                PrintTools.printlnStatus(3, tag, "  PRAGMA =",curr_ranges);
                // Examine "constraint" data for tighter ranges.
                // Use of constraint is safe only if there is one predecessor.
                if (curr_ranges != null && node.getPreds().size() == 1) {
                    curr_ranges.intersectRanges(
                            extractRangesFromConstraint(curr_ranges, node));
                }
                PrintTools.printlnStatus(3, tag, "  CONSTRAINT =", curr_ranges);
            }
            // Detected changes trigger propagation
            if (prev_ranges == null || !prev_ranges.equals(curr_ranges)) {
                if (node != g.getEntry()) {
                    // Keep the IPA result for the entry node.
                    node.putData("ranges", curr_ranges);
                }
                // Apply state changes due to the execution of the node.
                updateRanges(node);
                // Clean up after exiting a scope.
                exitScope(node);
                // Process successors.
                for (DFANode succ : node.getSuccs()) {
                    // Do not add successors for infeasible paths
                    if (succ.getPredData(node) != null) {
                        work_list.put((Integer)succ.getData("top-order"), succ);
                        PrintTools.printlnStatus(3, tag, "  OUT#",
                                succ.getData("top-order"), "=",
                                succ.getPredData(node));
                    }
                }
            }
        }
    }

    // Add intialized values from the declarations.
    private static void enterScope(DFANode node, RangeDomain ranges) {
        SymbolTable st = node.getData("symbol-entry");
        if (st == null) {
            return;
        }
        for (Symbol var : SymbolTools.getSymbols(st)) {
            // Not interested in other declarators.
            if (!(var instanceof VariableDeclarator)) {
                continue;
            }
            Initializer init = ((VariableDeclarator)var).getInitializer();
            // Flow of initializer is not guaranteed.
            if (IRTools.containsSideEffect(init)) {
                ranges.clear();
                return;
            }
            // Extract only scalar integers' initial values.
            if (init == null ||
                init.getChildren().size() != 1 ||
                !SymbolTools.isScalar(var) ||
                !isTractableType(var) ||
                SymbolTools.containsSpecifier(var, Specifier.VOLATILE) ||
                SymbolTools.containsSpecifier(var, Specifier.STATIC)) {
                continue;
            }
            Expression new_range =
                    Symbolic.simplify((Expression)init.getChildren().get(0));
            if (isTractableRange(new_range)) {
                ranges.setRange(var, new_range);
            }
        }
    }

    // Clean up variables that are not present in the scope.
    private static void exitScope(DFANode node) {
        List<SymbolTable> symbol_exits = node.getData("symbol-exit");
        if (symbol_exits != null) {
            // symbol-exit assumed to be list of symbol tables
            for (SymbolTable st : symbol_exits) {
                for (DFANode succ : node.getSuccs()) {
                    RangeDomain ranges_out = succ.getPredData(node);
                    if (ranges_out != null) {
                        ranges_out.removeSymbols(SymbolTools.getSymbols(st));
                    }
                }
            }
        }
    }

    // Check if the expression is tractable by range analysis.
    private static boolean isTractableRange(Expression e) {
        DFIterator<Expression> iter =
                new DFIterator<Expression>(e, Expression.class);
        while (iter.hasNext()) {
            Expression child = iter.next();
            if (child instanceof IntegerLiteral) {
                long value = ((IntegerLiteral)child).getValue();
                if (value >= Integer.MAX_VALUE/4 ||
                    value <= Integer.MIN_VALUE/4) {
                    return false;
                    // Avoid overflow from potentially large numbers.
                }
            }
            if (!tractable_class.contains(child.getClass())) {
                return false;
            }
            if (child instanceof Identifier) {
                Identifier id = (Identifier)child;
                if (!isTractableType(id.getSymbol())) {
                    return false;
                }
            } else if (child instanceof BinaryExpression &&
                       !tractable_op.contains(
                            ((BinaryExpression)child).getOperator())) {
                return false;
            } else if (child instanceof UnaryExpression &&
                       !tractable_op.contains(
                            ((UnaryExpression)child).getOperator())) {
                return false;
            }
            // Other cases should be all tractable.
        }
        return true;
    }

    // Check if the symbol is tractable by range analysis.
    protected static boolean isTractableType(Symbol symbol) {
        return (isInteger(symbol) &&
                !SymbolTools.containsSpecifier(symbol, Specifier.UNSIGNED));
    }

    // Detect invertible assignment and return the inverted expression.
    private static Expression
            invertExpression(Expression to, Expression from) {
        Symbol var = SymbolTools.getSymbolOf(to);
        Expression diff = Symbolic.subtract(to, from);
        if (IRTools.containsSymbol(diff, var)) {
            return null;
        }
        return Symbolic.add(to, diff);
    }

    // Methods for updating edges to successors.
    private static void updateRanges(DFANode node) {
        Object o = CFGraph.getIR(node);
        if (o instanceof ExpressionStatement) {
            o = ((ExpressionStatement)o).getExpression();
        }
        // Side-effect node
        if (o instanceof Traversable) {
            Traversable t = (Traversable)o;
            if (IRTools.containsClass(t, VaArgExpression.class)) {
                updateUnsafeNode(node);
                return;
            }
            if (IRTools.containsClass(t, FunctionCall.class)) {
                if (ip_node == null) {
                    updateFunctionCall(node);
                } else {
                    updateFunctionCallWithIPA(node);
                }
                return;
            }
        }
        // Assignments
        if (o instanceof AssignmentExpression) {
            updateAssignment(node, (AssignmentExpression)o);
        } else if (o instanceof BinaryExpression) {
            // Binary expressions (logical expression)
            updateBinary(node, (BinaryExpression)o);
        } else if (o instanceof SwitchStatement) {
            // Switch statements
            updateSwitch(node, (SwitchStatement) o);
        } else {
            // Side-effect-free node
            updateSafeNode(node);
        }
    }

    // Node containing function calls.
    private static void updateFunctionCall(DFANode node) {
        Traversable t = (Traversable)CFGraph.getIR(node);
        DFIterator<FunctionCall> iter =
                new DFIterator<FunctionCall>(t, FunctionCall.class);
        while (iter.hasNext()) {
            FunctionCall fc = iter.next();
            if (!StandardLibrary.isSideEffectFree(fc) &&
                !safe_functions.contains(fc.getName().toString())) {
                updateUnsafeNode(node);
                return;
            }
        }
        RangeDomain range =
                new RangeDomain((RangeDomain)node.getData("ranges"));
        for (Symbol def : DataFlowTools.getDefSymbol(t)) {
            range.removeRangeWith(def);
        }
        for (DFANode succ : node.getSuccs()) {
            succ.putPredData(node, range);
        }
    }

    // Returns a set of symbols whose addresses were taken, hence possibly
    // unsafe to have a value range.
    private static Set<Symbol> getAddressTakenID(Traversable tr) {
        Set<Symbol> ret = new HashSet<Symbol>();
        DFIterator<UnaryExpression> iter =
                new DFIterator<UnaryExpression>(tr, UnaryExpression.class);
        while (iter.hasNext()) {
            UnaryExpression ue = iter.next();
            if (ue.getExpression() instanceof Identifier &&
                ue.getOperator() == UnaryOperator.ADDRESS_OF) {
                ret.add(SymbolTools.getSymbolOf(ue.getExpression()));
            }
        }
        return ret;
    }

    // Update function call's ranges using the interprocedural input.
    @SuppressWarnings("unchecked")
    private static void updateFunctionCallWithIPA(DFANode node) {
        Map<CallSite, Domain> maymods = ip_node.getData(MayMod.tag + "CallOUT");
        Domain range = new RangeDomain((RangeDomain)node.getData("ranges"));
        Domain fc_gen = null;
        Traversable tr = (Traversable)CFGraph.getIR(node);
        // Kill ranges due to visible modifications in the node.
        range.kill(DataFlowTools.getDefSymbol(tr));
        // Kill ranges due to side effects.
        // Collect new ranges returning from the callee.
        DFIterator<FunctionCall> iter =
                new DFIterator<FunctionCall>(tr, FunctionCall.class);
        while (iter.hasNext()) {
            FunctionCall fc = iter.next();
            CallSite call_site = ip_node.getCallSite(fc);
            Domain maymod = maymods.get(call_site);
            if (maymod instanceof SetDomain) {
                range.kill((Set<Symbol>)maymod);
            } else if (StandardLibrary.isSideEffectFreeExceptIO(fc)) {
                ;
            } else {
                ((RangeDomain)range).killGlobal();
            }
            // Additionally kill symbols whose address is taken.
            range.kill(getAddressTakenID(fc));
            if (fc_gen == null) {
                fc_gen = call_site.out();
            } else {
                fc_gen = fc_gen.union(call_site.out());
            }
        }
        range = range.intersect(fc_gen);
        for (DFANode succ : node.getSuccs()) {
            succ.putPredData(node, range);
        }
    }

    // Side-effect-present node.
    private static void updateUnsafeNode(DFANode node) {
        for (DFANode succ : node.getSuccs()) {
            succ.putPredData(node, new RangeDomain());
        }
    }

    // Side-effect-free node.
    private static void updateSafeNode(DFANode node) {
        RangeDomain ranges_in = node.getData("ranges");
        for (DFANode succ : node.getSuccs()) {
            succ.putPredData(node, new RangeDomain(ranges_in));
        }
    }

    // Update assignments.
    private static void
            updateAssignment(DFANode node, AssignmentExpression e) {
        // Compute invariant information and cache it.
        if ((Integer)node.getData("num-visits") < 2) {
            Expression to = e.getLHS();
            Expression from = Symbolic.simplify(e.getRHS());
            String direction = "nochange";
            // Case 1. Dereference is treated conservatively.
            if (IRTools.containsUnary(to, UnaryOperator.DEREFERENCE)) {
                List types = SymbolTools.getExactExpressionType(to);
                if (types == null) {
                    direction = "discard";
                } else if (types.size() == 1 &&
                           (types.get(0) == Specifier.INT ||
                            types.get(0) == Specifier.LONG)) {
                    // starts from conservative direction.
                    direction = "discard";
                    Symbol var = SymbolTools.getSymbolOf(to);
                    if (var != null) {
                        Statement stmt = node.getData("stmt-ref");
                        if (stmt == null) {
                            stmt = e.getStatement();
                        }
                        if (stmt != null) {
                            Set alias_set = alias.get_alias_set(stmt, var);
                            if (alias_set != null
                                && !alias_set.contains("*")) {
                                direction = "kill";
                                node.putData("kill-set", alias_set);
                            }
                        }
                    }
                } else {
                    // assignments to non-integer variables.
                    direction = "nochange";
                }
            } else if (to instanceof Identifier) {
                Symbol var = ((Identifier)to).getSymbol();
                // Case 1. Unknown types are treated conservatively.
                if (var == null) {
                    direction = "discard";
                } else if (!isTractableType(var)) {
                // Case 2. Lvalue is not tractable by range analysis.
                    direction = "nochange";
                } else if (!isTractableRange(from)) {
                // Case 3. Lvalue is tractable but range is not.
                    direction = "kill";
                } else if (!IRTools.containsSymbol(from, var)) {
                // Case 4. Typical assignment with no self dependence.
                    direction = "normal";
                } else {
                    Expression inverted = invertExpression(to, from);
                    if (inverted == null) {
                    // Case 5. There is a self dependence, and it is not
                    // invertible.
                        direction = "recurrence";
                    } else if (IRTools.containsClass(
                            inverted, ArrayAccess.class)) {
                    // Case 5.5. Inverted expression contains array access.
                        direction = "kill";
                    } else {
                    // Case 6. There is a self dependence, and it is invertible.
                        direction = "invertible";
                        node.putData(direction, inverted);
                    }
                }
                node.putData("assign-to", to);
                node.putData("assign-from", from);
            } else {
                node.putData("assign-to", to);
                direction = "kill";
            }
            // Case 7. Lvalue is not simple; no source of information.
            node.putData("assign-direction", direction);
        }

        RangeDomain ranges_in = node.getData("ranges");
        RangeDomain ranges_out = new RangeDomain(ranges_in);
        String direction = node.getData("assign-direction");
        Object node_data = null;

        if (direction.equals("discard")) {
            // Dicards everything after this node.
            ranges_out = new RangeDomain();
        } else if (direction.equals("kill") &&
                   node.getData("kill-set") != null) {
            // Cases having aliased kill.
            Set kill_set = node.getData("kill-set");
            for (Object var : kill_set) {
                if (var instanceof Symbol) {
                    ranges_out.removeRangeWith((Symbol) var);
                }
            }
        } else if (!((node_data = node.getData("assign-to"))
                    instanceof Identifier)) {
            // Kills ranges containing non-identifiers that modified at this
            // node.
            ranges_out.removeRangeWith(
                    SymbolTools.getSymbolOf((Expression)node_data));
        } else if (!direction.equals("nochange")) {
            // Cases where identifiers are modified.
            Symbol var = ((Identifier)node.getData("assign-to")).getSymbol();
            // Preprocess the range to avoid replacement in array subscripts.
            ranges_out.killArraysWith(var);
            Expression replace_with =
                (direction.equals("invertible")) ?
                    (Expression)node.getData("invertible") :
                    ranges_out.getRange(var);
            Expression from = node.getData("assign-from");
            // Expand expressions that contain the killed symbol.
            if (direction.equals("invertible") ||
                direction.equals("recurrence")) {
                from = ranges_out.expandSymbol(from, var);
                ranges_out.expandSymbol(var);
            }
            // Eliminate the assigned symbol in the range.
            ranges_out.replaceSymbol(var, replace_with);
            // Postprocess the range by discarding cyclic ranges.
            ranges_out.removeRecurrence();
            // Remove or keep the range for the assigned symbol.
            if (direction.equals("kill")) {
                ranges_out.removeRange(var);
            } else {
                ranges_out.setRange(var, from);
            }
            // Add additional ranges from the equality condition -- disable if
            // there is any problem.
            /*
               Expression eq_expr = Symbolic.eq(e.getLHS(), e.getRHS());
               RangeDomain eq_range = extractRanges(eq_expr);
               eq_range.removeRange(var);
               for ( Symbol eq_var : eq_range.getSymbols() )
               if ( ranges_out.getRange(eq_var) == null )
               ranges_out.setRange(eq_var, eq_range.getRange(eq_var));
             */
        }
        // Update successors.
        for (DFANode succ : node.getSuccs()) {
            succ.putPredData(node, ranges_out);
        }
    }

    // Apply conditional expressions.
    private static void updateBinary(DFANode node, BinaryExpression e) {
        // Adjustment for unexpanded expression due to may-be evaluated
        // expression. Only safe evaluation is taken out of any expressions
        // used in conditional branches, and there should be a checking
        // mechanism that detects unsafe evaluations.
        if (IRTools.containsSideEffect(e)) {
            updateUnsafeNode(node);
            return;
        } else if (node.getSuccs().size() < 2) {
        // This is not a branch; residue of normalization.
            updateSafeNode(node);
            return;
        }
        // Compute or retrieve the ranges from the conditional expression.
        if ((Integer)node.getData("num-visits") < 2) {
            DFANode true_succ = node.getData("true");
            DFANode false_succ = node.getData("false");
            Expression negated = Symbolic.negate(e);
            RangeDomain true_range = extractRanges(e);
            RangeDomain false_range = extractRanges(negated);
            if (option == RANGE_PRAGMA) {
                true_succ.putData("constraint", e);
                false_succ.putData("constraint", negated);
            }
            node.putSuccData(true_succ, true_range);
            node.putSuccData(false_succ, false_range);
        }
        RangeDomain ranges_in = node.getData("ranges");
        for (DFANode succ : node.getSuccs()) {
            RangeDomain ranges_out = new RangeDomain(ranges_in);
            ranges_out.intersectRanges((RangeDomain)node.getSuccData(succ));
            // Infeasible path detected, so remove the data
            if (ranges_in.size() > ranges_out.size()) {
                ranges_out = null;
            }
            succ.putPredData(node, ranges_out);
        }
    }

    // Apply conditional expressions from switch statement.
    private static void updateSwitch(DFANode node, SwitchStatement s) {
        Expression lhs = s.getExpression();
        if (IRTools.containsSideEffect(lhs)) {
            updateUnsafeNode(node);
            return;
        }
        // Compute and cache the extracted ranges. 
        if ((Integer)node.getData("num-visits") < 2) {
            for (DFANode succ : node.getSuccs()) {
                Object o = CFGraph.getIR(node);
                if (o instanceof Case) {
                    node.putSuccData(succ, extractRanges(
                        Symbolic.eq(lhs, ((Case)o).getExpression())));
                }
            }
        }
        RangeDomain ranges_in = node.getData("ranges");
        for (DFANode succ : node.getSuccs()) {
            RangeDomain ranges_out = new RangeDomain(ranges_in);
            RangeDomain edge_range = (RangeDomain)node.getSuccData(succ);
            if (edge_range != null) {
                ranges_out.intersectRanges(edge_range);
            }
            // Infeasible path detected, so remove the data
            if (ranges_in.size() > ranges_out.size()) {
                ranges_out = null;
            }
            succ.putPredData(node, ranges_out);
        }
    }

    /**
    * Extracts ranges if there exist a {@code constraint} information stored in
    * the given node.
    * @param ranges the input ranges.
    * @param node the node to be analyzed.
    * @return the resulting range.
    */
    private static RangeDomain
            extractRangesFromConstraint(RangeDomain ranges, DFANode node) {
        RangeDomain ret = new RangeDomain();
        Expression constraint = node.getData("constraint");
        if (constraint == null) {
            return ret;
        }
        for (Symbol symbol : ranges.getSymbols()) {
            Expression range = ranges.getRange(symbol);
            if (range instanceof RangeExpression) {
                RangeDomain temp = new RangeDomain();
                temp.setRange(symbol, range);
                Expression intersect =
                        Symbolic.and(constraint, temp.toExpression());
                ret.intersectRanges(extractRanges(intersect));
            }
        }
        return ret;
    }

    /**
    * Extracts ranges provided by user-inserted pragma. It is assumed the pragma
    * is a type of {@link cetus.hir.PragmaAnnotation.Range}, placed right before
    * the target statement.
    * @param node the node to be analyzed.
    * @return the resulting ranges combined with the input and the pragma.
    */
    private static RangeDomain extractRangePragma(DFANode node) {
        RangeDomain ret = new RangeDomain();
        if (node.getData("stmt") == null ||
            !(node.getData("stmt") instanceof Statement)) {
            return ret;
        }
        Statement stmt = node.getData("stmt");
        List<PragmaAnnotation.Range> range_pragmas =
                stmt.getAnnotations(PragmaAnnotation.Range.class);
        for (PragmaAnnotation.Range range_pragma : range_pragmas) {
            RangeDomain rd = new RangeDomain();
            Map<Identifier, RangeExpression> range_map = range_pragma.getMap();
            for (Identifier id : new HashSet<Identifier>(range_map.keySet())) {
                // Checks if the user-inserted expression contains valid
                // symbols at the program point.
                Identifier valid_id =
                        (Identifier)getValidExprFromPragma(id, stmt);
                Expression valid_range =
                        getValidExprFromPragma(range_map.get(id), stmt);
                if (valid_id == null || valid_range == null) {
                    PrintTools.printlnStatus(0, tag,
                            "Skipping ineligible range pragma item:",
                            id, "=", range_map.get(id));
                    range_map.remove(id);
                    continue;
                }
                rd.setRange(valid_id.getSymbol(), valid_range);
            }
            ret.intersectRanges(rd);
        }
        if (ret.size() > 0) {
            ret = extractRanges(ret.toExpression());
        }
        return ret;
    }

    /**
    * Returns a valid expression from the given possibly {@code invalid}
    * expression read from pragma after linking the appropriate symbols and
    * checking for eligible expression type.
    * @param e the given possibly invalid expression.
    * @param stmt the reference point where symbol is searched.
    * @return the resulting expression, null if it is not valid.
    */
    private static Expression
            getValidExprFromPragma(Expression e, Statement stmt) {
        Expression ret = e.clone();
        DFIterator<Identifier> iter =
                new DFIterator<Identifier>(ret, Identifier.class);
        while (iter.hasNext()) {
            Identifier id = iter.next();
            Symbol symbol = SymbolTools.getSymbolOfName(id.getName(), stmt);
            if (symbol == null || !isInteger(symbol)) {
                return null;
            }
            Identifier valid_id = new Identifier(symbol);
            if (e instanceof Identifier) {
                return valid_id;
            } else {
                valid_id.swapWith(id);
            }
        }
        return ret;
    }

    // Extract a set of ranges from the preprocessed list
    private static RangeDomain extractRanges(List solved) {
        RangeDomain ret = new RangeDomain();
        for (Object o : solved) {
            if (!(o instanceof BinaryExpression))
                continue;
            BinaryExpression be = (BinaryExpression)o;
            BinaryOperator op = be.getOperator();
            Expression lhs = be.getLHS();
            Expression rhs = Symbolic.simplify(be.getRHS());
            Symbol var = ((Identifier)lhs).getSymbol();
            if (!(lhs instanceof Identifier) ||
                !isTractableType(var) ||
                !isTractableRange(rhs)) {
                continue;
            }
            if (op == BinaryOperator.COMPARE_EQ) {
                ret.setRange(var, rhs.clone());
            } else if (op == BinaryOperator.COMPARE_GE) {
                ret.setRange(var,
                             new RangeExpression(Symbolic.simplify(rhs),
                                                 new InfExpression(1)));
            } else if (op == BinaryOperator.COMPARE_GT) {
                ret.setRange(var,
                             new RangeExpression(Symbolic.add(rhs, one),
                                                 new InfExpression(1)));
            } else if (op == BinaryOperator.COMPARE_LE) {
                ret.setRange(var,
                             new RangeExpression(new InfExpression(-1),
                                                 Symbolic.simplify(rhs)));
            } else if (op == BinaryOperator.COMPARE_LT) {
                ret.setRange(var,
                             new RangeExpression(new InfExpression(-1),
                                                 Symbolic.subtract(rhs, one)));
            }
        }
        return ret;
    }

    // Extract a set of ranges from the given binary expression
    public static RangeDomain extractRanges(Expression e) {
        RangeDomain ret = new RangeDomain();
        Expression simp = Symbolic.simplify(e);
        if (!(simp instanceof BinaryExpression)) {
            return ret;
        }
        BinaryExpression be = (BinaryExpression)simp;
        List solved = Symbolic.getVariablesOnLHS(be);
        BinaryOperator op = be.getOperator();
        if (op == BinaryOperator.LOGICAL_AND) {
            for (Object o : solved) {
                ret.intersectRanges(extractRanges((List)o));
            }
        } else if (op == BinaryOperator.LOGICAL_OR) {
            for (Object o : solved) {
                ret.unionRanges(extractRanges((List)o));
            }
        } else {
            ret = extractRanges(solved);
        }
        return ret;
    }

    /**
    * Returns a range domain associated with the specified statement. It will
    * return an already existing range domain if one is found. Otherwise,
    * the range domains for the procedure containing {@code stmt} are recomputed
    * , and then the associated range domain is returned.
    * @param stmt the statement of interest.
    * @return the associated range domain.
    */
    public static RangeDomain query(Statement stmt) {
        RangeDomain ret = range_domains.get(stmt);
        if (ret == null) {
            Procedure proc = stmt.getProcedure();
            if (proc == null) {
                PrintTools.printlnStatus(0, tag, "[WARNING]",
                        "Range cannot be retrieved for an orphan statement.");
                ret = empty_range;
            } else {
                PrintTools.printlnStatus(1, tag,
                        "Invoking range analysis for", proc.getName());
                range_domains.putAll(getRanges(proc));
                ret = range_domains.get(stmt);
            }
        }
        return ret;
    }

    /**
    * Invalidates range domains stored in the static spaces. Every IR-changing
    * passes need to call this method.
    */
    public static void invalidate() {
        range_domains.clear();
    }

    /**
    * Register the specified function names in the safe list of function calls.
    * During range analysis these functions are considered to have no side
    * effects at all.
    */
    public static void registerSafeFunction(String name, String... names) {
        safe_functions.add(name);
        for (String n : names) {
            safe_functions.add(n);
        }
    }
}
