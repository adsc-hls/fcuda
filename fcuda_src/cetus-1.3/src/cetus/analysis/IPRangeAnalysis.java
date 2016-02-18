package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
* Class IPRangeAnalysis performs interprocedural range analysis.
*/
public class IPRangeAnalysis extends IPAnalysis {

    /** Set of specifiers to be handled in the analysis */
    private static Set<Specifier> include;

    /** Set of specifiers not to be handled in the analysis */
    private static Set<Specifier> exclude;

    static {
        include = new HashSet<Specifier>();
        include.add(Specifier.INT);
        include.add(Specifier.LONG);
        include.add(Specifier.SIGNED);
        exclude = new HashSet<Specifier>();
        exclude.add(Specifier.UNSIGNED);
        exclude.add(Specifier.CHAR);
        exclude.add(Specifier.STATIC);
        exclude.add(Specifier.EXTERN);
        exclude.add(PointerSpecifier.UNQUALIFIED);
        exclude.add(PointerSpecifier.CONST);
        exclude.add(PointerSpecifier.VOLATILE);
        exclude.add(PointerSpecifier.CONST_VOLATILE);
    }
    
    /** Pass name */
    private static final String tag = "[IPA:Range]";

    /**
    * Drives IPA by calling a customized MayMod analysis.
    */
    public static void compute(Program program) {
        MayMod maymod = new MayMod(program);
        maymod.setFilter(include, exclude);
        maymod.start();
        maymod.saveData();
        maymod.callgraph.clean();
        IPRangeAnalysis iprange = new IPRangeAnalysis(maymod.callgraph);
        iprange.start();
        PrintTools.printlnStatus(1,
                tag, "Report", PrintTools.line_sep, iprange.getReport());
    }

    /**
    * Constructs a IPRangeAnalysis object with the given input program.
    */
    private IPRangeAnalysis(Program program) {
        super(program, Option.FORWARD, Option.BACKWARD, Option.BOTTOMUP,
              Option.WORKLIST, Option.FAILFAST);
        name = tag;
    }

    /**
    * Constructs a IPRangeAnalysis object with the given callgraph
    * representation.
    */
    private IPRangeAnalysis(IPAGraph callgraph) {
        super(callgraph, Option.FORWARD, Option.BACKWARD, Option.BOTTOMUP,
              Option.WORKLIST, Option.FAILFAST);
        name = tag;
    }

    public void start() {
        super.start();
        Map<Procedure, Map<Statement, RangeDomain>> ip_ranges =
                new LinkedHashMap<Procedure, Map<Statement, RangeDomain>>();
        Iterator<IPANode> iter = callgraph.topiterator();
        while (iter.hasNext()) {
            IPANode node = iter.next();
            Map<Statement, RangeDomain> map = RangeAnalysis.getRanges(node);
            ip_ranges.put(node.getProcedure(), map);
        }
        RangeAnalysis.setRanges(ip_ranges);
    }

    /**
    * Removes any existing interprocedural results.
    */
    public static void clear() {
        RangeAnalysis.setRanges(null);
    }

    /**
    * Performs intraprocedural range analysis.
    */
    public void analyzeProcedure(IPANode node) {
        CFGraph result = RangeAnalysis.getRangeCFG(node);
        node.putData("cfg", result);
    }

    /**
    * Detects any changes in the range domain before call sites.
    * Detection is made on the pre-selected version of the range domain whose
    * content is accessible and valid in the callee's context.
    */
    public Set<CallSite> updateCalls(IPANode node) {
        Set<CallSite> ret = new LinkedHashSet<CallSite>();
        CFGraph cfg = node.getData("cfg");
        for (CallSite call_site : node.getCallSites()) {
            FunctionCall fc = call_site.getFunctionCall();
            DFANode fc_node = cfg.getCallNode(fc);
            if (fc_node == null) {
                continue;
                // function calls in initializers are handled conservatively.
            }
            RangeDomain range = fc_node.getData("ranges");
            Domain curr_data = (range == null) ?
                    NullDomain.getNull() : range.clone();
            IPANode callee = call_site.getCallee();
            if (curr_data instanceof RangeDomain) {
                RangeDomain curr_range = (RangeDomain)curr_data;
                // Forward substitute at the call site for maximum exposure to
                // callees.
                curr_range.substituteForwardRange();
                if (callee == null) {
                    Set<Symbol> fc_vars = SymbolTools.getAccessedSymbols(fc);
                    curr_range.killGlobalAnd(fc_vars);
                    // remove globals and function call arguments.
                } else if (callee.containsVarArg()) {
                    curr_range.killLocal();
                } else {
                    curr_range.killLocalExcept(new LinkedHashSet<Symbol>(
                            SymbolTools.exprsToSymbols(
                                    call_site.getIDArguments())));
                } // TODO: multiple function call handling is missing now.
            }
            if (!(call_site.in().equals(curr_data))) {
                ret.add(call_site);
                call_site.in(curr_data);
                if (callee != null && !callee.containsVarArg()) {
                    Domain callee_data = renameAndExtract(call_site, curr_data);
                    callee.in(call_site, callee_data);
                    // Check if no-clone is ok.
                }
            }
        }
        return ret;
    }

    private Domain renameAndExtract(CallSite call_site, Domain data) {
        if (data instanceof NullDomain) {
            return data;
        }
        // data should be a range domain.
        RangeDomain ranges = (RangeDomain)data;
        RangeDomain ret = new RangeDomain();
        List<Expression> id_args = call_site.getIDArguments();
        for (Symbol var : ranges.getSymbols()) {
            // Rename ranges
            Expression range = ranges.getRange(var).clone();
            for (Expression id_arg : id_args) {
                Symbol id_symbol = SymbolTools.getSymbolOf(id_arg);
                if (IRTools.containsSymbol(range, id_symbol)) {
                    Expression param_id = new Identifier(
                            call_site.argumentToParameter(id_arg));
                    range = IRTools.replaceSymbol(
                            range, SymbolTools.getSymbolOf(id_arg), param_id);
                    if (IRTools.containsSymbol(range, id_symbol)) {
                        range = null;
                        // couldn't remove arguments; just kill the range.
                        break;
                    }
                }
            }
            if (range == null) {
                continue;
            }
            // Rename variable keys
            Expression var_id = new Identifier(var);
            if (id_args.contains(var_id)) {
                ret.setRange(call_site.argumentToParameter(var_id), range);
            } else {
                ret.setRange(var, range);
            }
        }
        PrintTools.printlnStatus(5, tag, "Renamed", data, "=>", ret);
        // Extract ranges from the argument-parameter pairs.
        // For now, only extracts literal constants.
        for (Symbol param : call_site.getParameters()) {
            if (param.getSymbolName().equals("")) {
                continue;       // void symbol.
            }
            Expression matching_arg = call_site.parameterToArgument(param);
            if (matching_arg != null && matching_arg instanceof IntegerLiteral){
                ret.setRange(param, matching_arg.clone());
            }
        }
        PrintTools.printlnStatus(5, tag, "Extracted call ranges", "=>", ret);
        return ret;
    }

    // Not used.
    public boolean updateCall(IPANode node) {
        return (updateCalls(node).size() > 0);
    }

    /**
    * Checks any data change due to return from callee and updates the OUT data
    * if applicable.
    */
    public boolean updateReturn(IPANode node) {
        boolean ret = false;
        CFGraph cfg = node.getData("cfg");
        List<DFANode> exits = cfg.getExitNodes();
        Domain curr_data = null;
        for (DFANode exit : exits) {
            RangeDomain range = exit.getData("ranges");
            Domain data = null;
            if (range == null) {
                data = NullDomain.getNull();
            } else {
                data = range.clone();
                data.kill(DataFlowTools.getDefSymbol(
                        (Traversable)CFGraph.getIR(exit)));
            }
            if (data instanceof RangeDomain) {
                ((RangeDomain)data).substituteForwardRange();
                ((RangeDomain)data).killLocal();
                ((RangeDomain)data).killOrphan();
            }
            if (curr_data == null) {
                curr_data = data;
            } else {
                curr_data = curr_data.union(data);
            }
        }
        PrintTools.printlnStatus(5, tag, node.getName(), "OUT =", curr_data);
        if (!curr_data.equals(node.out())) {
            ret = true;
            PrintTools.printlnStatus(5, tag,
                    "backward change detected", node.out(), "=>", curr_data);
            node.out(curr_data);
            for (CallSite calling_site : node.getCallingSites()) {
                calling_site.out(curr_data);
            }
        }
        return ret;
    }

    private int countGlobals() {
        int ret = 0;
        List<Traversable> children = program.getChildren();
        for (int i = 0; i < children.size(); i++) {
            TranslationUnit tu = (TranslationUnit)children.get(i);
            for (Symbol var : SymbolTools.getSymbols(tu)) {
                if (SymbolTools.isArray(var)) {
                    continue;
                }
                for (Object type : var.getTypeSpecifiers()) {
                    if (include.contains(type) && !exclude.contains(type)) {
                        ret++;
                    }
                }
            }
        }
        return ret;
    }

    private int countFormals() {
        int ret = 0;
        Iterator<IPANode> iter = callgraph.topiterator();
        while (iter.hasNext()) {
            for (Object o : iter.next().getProcedure().getParameters()) {
                VariableDeclaration vdecln = (VariableDeclaration)o;
                if (vdecln.getNumDeclarators() == 1) {
                    Symbol var = (Symbol)vdecln.getChildren().get(0);
                    for (Object type:var.getTypeSpecifiers()) {
                        if (include.contains(type) && !exclude.contains(type)) {
                            ret++;
                        }
                    }
                }
            }
        }
        return ret;
    }

    private int[] countForwards() {
        int[] ret = new int[2];
        ret[0] = 0;
        ret[1] = 0;
        Set<Symbol> forward_vars = new HashSet<Symbol>();
        Iterator<IPANode> iter = callgraph.topiterator();
        while (iter.hasNext()) {
            Domain in = iter.next().in();
            if (in instanceof RangeDomain) {
                forward_vars.addAll(((RangeDomain)in).getSymbols());
                ret[1] += ((RangeDomain) in).size();
            }
        }
        ret[0] = forward_vars.size();
        return ret;
    }

    private int[] countBackwards() {
        int[] ret = new int[2];
        ret[0] = 0;
        ret[1] = 0;
        Set<Symbol> backward_vars = new HashSet<Symbol>();
        Iterator<IPANode> iter = callgraph.topiterator();
        while (iter.hasNext()) {
          for (CallSite call_site:iter.next().getCallSites()) {
                Domain out = call_site.out();
                if (out instanceof RangeDomain) {
                    backward_vars.addAll(((RangeDomain)out).getSymbols());
                    ret[1] += ((RangeDomain) out).size();
                }
            }
        }
        ret[0] = backward_vars.size();
        return ret;
    }

    private String getReport() {
        int num_globals = countGlobals();
        int num_formals = countFormals();
        int[] num_forwards = countForwards();
        int[] num_backwards = countBackwards();
        String ret =
            "#Globals        = " + num_globals + "\n" +
            "#Formals        = " + num_formals + "\n" +
            "#ForwardVars    = " + num_forwards[0] + "\n" +
            "#ForwardRanges  = " + num_forwards[1] + "\n" +
            "#BackwardVars   = " + num_backwards[0] + "\n" +
            "#BackwardRanges = " + num_backwards[1] + "\n";
        return callgraph.getReport() + ret;
    }
}
