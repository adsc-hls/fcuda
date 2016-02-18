package cetus.analysis;

import cetus.hir.*;

import java.util.Iterator;
import java.util.Set;

/**
 * Class MayMod instantiates an IPA problem that computes may-modified
 * variables. It is able to collect the subset of the solution by specifying
 * "include" and "exclude" sets which are sets of type specifiers.
 */
public class MayMod extends IPAnalysis {

    private Set<Specifier> include, exclude;

    protected static final String tag = "[IPA:MayMod]";

    public MayMod(Program program) {
        super(program, Option.BACKWARD, Option.BOTTOMUP);
        name = tag;
    }
    
    public MayMod(IPAGraph callgraph) {
        super(callgraph.getProgram(), Option.BACKWARD, Option.BOTTOMUP);
        this.program = callgraph.getProgram();
        this.callgraph = callgraph;
        name = tag;
    }

    public void setFilter(Set<Specifier> include, Set<Specifier> exclude) {
        this.include = include;
        this.exclude = exclude;
    }

    public void analyzeProcedure(IPANode node) {
        Domain maymods = new SetDomain<Symbol>(
                DataFlowTools.getDefSymbol(node.getProcedure()));
        for (CallSite call_site : node.getCallSites()) {
            maymods = maymods.union(call_site.out());
        }
        if (maymods instanceof SetDomain) {
            // Select only global variables with the specified type constraints.
            Iterator iter = ((Set)maymods).iterator();
            while (iter.hasNext()) {
                Symbol var = (Symbol) iter.next();
                if (!SymbolTools.isGlobal(var)) {
                    iter.remove();
                    continue;
                }
                for (Object type : var.getTypeSpecifiers()) {
                    if (include != null && !include.contains(type) ||
                        exclude != null && exclude.contains(type)) {
                        iter.remove();
                        break;
                    }
                }
            }
        }
        // Put the current solution in the temporary space.
        node.putData(name, maymods);
    }

    // MayMod has no forward data along the call graph.
    public Set<CallSite> updateCalls(IPANode node) {
        return null;
    }

    public boolean updateCall(IPANode node) {
        return false;
    }

    public boolean updateReturn(IPANode node) {
        // Checks state changes.
        boolean ret = false;
        Domain prev_maymods = node.out();
        Domain curr_maymods = node.getData(name);
        // No change detected.
        if (prev_maymods.equals(curr_maymods)) {
            return ret;
        }
        // Update OUT data of the node and the calling sites.
        ret = true;
        node.out(curr_maymods);
        for (CallSite calling_site : node.getCallingSites()) {
            calling_site.out(curr_maymods);
        }
        return ret;
    }
}
