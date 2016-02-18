package cetus.analysis;

import cetus.hir.DFIterator;
import cetus.hir.FunctionCall;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.PrintTools;
import cetus.hir.Traversable;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Class IPAGraph provides a call graph for interprocedural analysis.
 */
public class IPAGraph extends DFAGraph {

    // Exceptions.
    private static final int CONTAINS_CYCLE = 1;
    private static final int CONTAINS_FUNCTION_POINTER = 2;

    // The program for this call graph.
    private Program program;

    // The root node of the call graph.
    private IPANode root;

    // Topological ordering of the call graph disregarding back edges.
    private ArrayList<IPANode> toporder;

    // The total number of call sites.
    private int num_callsites;

    // Exception code.
    private int exception;

    /**
    * Constructs a new IPAGraph with the given program and performs topological
    * ordering of the graph.
    */
    public IPAGraph(Program prog) {
        super();
        root = null;
        program = prog;
        toporder = new ArrayList<IPANode>();
        num_callsites = 0;
        exception = 0;
        buildGraph(prog);
        identifyCloneableNodes();
        buildTopOrder();
    }

    /**
    * Returns a string dump of the call graph.
    */
    @Override
    public String toString() {
        return toString("");
    }

    /**
    * Returns a string dump of the call graph with the given tag.
    * @param tag the keyword being printed for each node -- usually the name of
    * the analysis pass using this graph).
    * @return the string dump.
    */
    public String toString(String tag) {
        StringBuilder str = new StringBuilder(80);
        str.append("[IPAGraph] #ReachableNode = ").append(toporder.size());
        str.append("  #CallSites = ").append(num_callsites);
        str.append(PrintTools.line_sep);
        for (IPANode node : toporder) {
            str.append(node).append(PrintTools.line_sep);
        }
        return str.toString();
    }

    /**
    * Returns the ordered node number for the given node.
    */
    public Integer getTopOrder(IPANode node) {
        return toporder.indexOf(node);
    }

    /**
    * Returns the node having the given order.
    */
    public IPANode getNode(int order) {
        if (order >= 0 && order < toporder.size()) {
            return toporder.get(order);
        } else {
            return null;        // unreachable node or library calls.
        }
    }

    /**
    * Returns the node representing the given procedure.
    */
    public IPANode getNode(Procedure procedure) {
        for (DFANode node : nodes) {
            IPANode inode = (IPANode)node;
            if (inode.getProcedure() == procedure) {
                return inode;
            }
        }
        return null;
    }

    /**
    * Returns the firs node in the ordered list.
    */
    public IPANode getFirstNode() {
        return toporder.get(0);
    }

    /**
    * Returns the root node of the graph.
    */
    public IPANode getRoot() {
        return root;
    }

    /**
    * Returns the last node in the ordered list.
    */
    public IPANode getLastNode() {
        return toporder.get(toporder.size() - 1);
    }

    /**
    * Returns the last order in the ordered list.
    */
    public int getLastOrder() {
        return toporder.size() - 1;
    }

    public Iterator<IPANode> topiterator() {
        return toporder.iterator();
    }

    /**
    * Returns the program associated with this graph.
    */
    public Program getProgram() {
        return program;
    }

    /**
    * Cleans up the data stored in the graph.
    */
    public void clean() {
        Iterator<IPANode> iter = topiterator();
        while (iter.hasNext()) {
            iter.next().clean();
        }
    }

    public boolean containsFunctionPointer() {
        return ((exception & CONTAINS_FUNCTION_POINTER) != 0);
    }

    public boolean containsCycle() {
        return ((exception & CONTAINS_CYCLE) != 0);
    }

    // Returns the node associated with the given procedure.
    private IPANode checkNode(Procedure proc) {
        IPANode ret = getNode(proc);
        if (ret == null) {
            ret = new IPANode(proc);
            addNode(ret);
        }
        return ret;
    }

    // Builds a call graph with the given program.
    private void buildGraph(Program prog) {
        DFIterator<Traversable> iter = new DFIterator<Traversable>(prog);
        IPANode caller = null;
        while (iter.hasNext()) {
            Traversable o = iter.next();
            if (o instanceof Procedure) {
                caller = checkNode((Procedure)o);
                if (caller.getName().equals("main")) {
                    root = caller;
                    caller.setRoot();
                }
            } else if (o instanceof FunctionCall) {
                FunctionCall fcall = (FunctionCall)o;
                Procedure callee_proc = fcall.getProcedure();
                IPANode callee = null;
                if (callee_proc != null) {
                    callee = checkNode(callee_proc);
                    addEdge(caller, callee);
                }
                CallSite callsite =
                        new CallSite(++num_callsites, fcall, caller, callee);
                caller.getCallSites().add(callsite);
                if (callsite.containsFunctionPointer()) {
                    exception |= CONTAINS_FUNCTION_POINTER;
                }
                if (callee != null) {
                    callee.getCallingSites().add(callsite);
                }
            }
        }
    }

    // Performs topological sorting and stores the result in an ordered list.
    private void buildTopOrder() {
        if (root == null) {
            return;
        }
        topologicalSort(root);
        Iterator<DFANode> iter = new TopIterator();
        while (iter.hasNext()) {
            toporder.add((IPANode)iter.next());
        }
    }

    // Checks if there is any node that is not cloneable.
    private void identifyCloneableNodes() {
        if (root == null) {
            return;
        }
        List scc_forest = getSCC(root);
        int scc_num = 0;
        for (Object o : scc_forest) {
            if (o instanceof List) {      // it represents SCC.
                exception |= CONTAINS_CYCLE;
                for (Object oo : (List)o) {
                    ((IPANode)oo).setCloneable(false);
                }
            }
        }
    }

    /** Returns statisitcs of the graph */
    public String getReport() {
        int num_reachable_nodes = 0;
        int num_known_callees = 0;
        int num_unknown_callees = 0;
        for (IPANode node : toporder) {
            num_reachable_nodes++;
            for (CallSite call_site : node.getCallSites()) {
                if (call_site.getCallee() == null) {
                    num_unknown_callees++;
                } else {
                    num_known_callees++;
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("#Procedures     = ").append(num_reachable_nodes);
        sb.append(PrintTools.line_sep);
        sb.append("#CallSites      = ").append(num_unknown_callees +
                                               num_known_callees);
        sb.append(PrintTools.line_sep);
        sb.append("#KnownCallees   = ").append(num_known_callees);
        sb.append(PrintTools.line_sep);
        sb.append("#UnknownCallees = ").append(num_unknown_callees);
        sb.append(PrintTools.line_sep);
        return sb.toString();
    }
}
