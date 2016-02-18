package cetus.analysis;

import cetus.hir.*;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.*;

/**
 * A static call graph for the program.
 */
public class CallGraph {

    public class Caller {

        private Statement callsite;

        private Procedure callingproc;

        public Caller() {
            callsite = null;
            callingproc = null;
        }
        
        public Caller(Statement l_stmt) {
            callsite = l_stmt;
            callingproc = l_stmt.getProcedure();
        }
        
        public Statement getCallSite() {
            return callsite;
        }

        public Procedure getCallingProc() {
            return callingproc;
        }

    }

    public class Node {

        private Procedure proc;

        private ArrayList<Caller> callers;

        private ArrayList<Procedure> callees;

        public Node(Procedure proc) {
            this.proc = proc;
            callers = new ArrayList<Caller>(1);
            callees = new ArrayList<Procedure>(1);
        }
        
        public void addCaller(Statement stmt) {
            callers.add(new Caller(stmt));
        }

        public void addCallee(Procedure proc) {
            callees.add(proc);
        }

        public List<Caller> getCallers() {
            return callers;
        }

        public List<Procedure> getCallees() {
            return callees;
        }

        public Procedure getProcedure() {
            return proc;
        }

    }

    private Node root;

    private HashMap<Procedure, Node> callgraph;

    /**
    * Creates a call graph for the program.
    * Assumes the graph is rooted at a procedure
    * called "main" or "MAIN__".  (The second is
    * used in f2c code.)
    *
    * @param program The program for which to create the call graph.
    */
    public CallGraph(Program program) {
        callgraph = new HashMap<Procedure, Node>();
        /* add nodes for every procedure */
        DFIterator<Procedure> iter =
                new DFIterator<Procedure>(program, Procedure.class);
        iter.pruneOn(Procedure.class);
        while (iter.hasNext()) {
            Procedure proc = iter.next();
            String proc_name = proc.getName().toString();
            Node node = new Node(proc);
            /* f2c code uses MAIN__ */
            if (proc_name.equals("main") || proc_name.equals("MAIN__")) {
                root = node;
            }
            callgraph.put(proc, node);
        }
        /* look for function calls and put them in the graph */
        DFIterator<FunctionCall> fc_iter =
                new DFIterator<FunctionCall>(program, FunctionCall.class);
        while (fc_iter.hasNext()) {
            FunctionCall call = fc_iter.next();
            Node currproc = null, calledproc = null;
            Procedure l_currproc = call.getStatement().getProcedure();
            Procedure l_calledproc = call.getProcedure();
            // l_calledproc will be null for system calls
            if (l_calledproc != null) {
                currproc = callgraph.get(l_currproc);
                if (currproc != null) {
                    currproc.addCallee(l_calledproc);
                }
                calledproc = callgraph.get(l_calledproc);
                if (calledproc != null) {
                    calledproc.addCaller(call.getStatement());
                }
            }
        }
    }

    public boolean callsSelf(Procedure proc) {
        Node n = callgraph.get(proc);
        if (n != null && n.getCallees().contains(proc)) {
            return true;
        } else {
            return false;
        }
    }

    /**
    * Access the graph directly as a hash map.
    *
    * @return a HashMap representing the graph.
    */
    public HashMap getCallGraph() {
        return callgraph;
    }

    /**
    * Access the root node of the graph.
    *
    * @return the root Node object.
    */
    public Node getRoot() {
        return root;
    }

    /**
    * Determines if the procedure is a leaf of the call graph.
    *
    * @param proc A Procedure appearing in the call graph.
    * @return true if the procedure does not call and other procedures of the
    *       program (i.e.,library calls are ignored), or false otherwise.
    */
    public boolean isLeaf(Procedure proc) {
        Node node = callgraph.get(proc);
        return (node.getCallees().size() == 0);
    }

    public boolean isRecursive(Procedure proc) {
        HashSet<Procedure> seen = new HashSet<Procedure>();
        HashSet<Procedure> horizon = new HashSet<Procedure>();
        horizon.add(proc);
        while (!horizon.isEmpty()) {
            Procedure p = horizon.iterator().next();
            Node n = callgraph.get(p);
            if (n != null && n.getCallees().contains(proc)) {
                return true;
            }
            horizon.remove(p);
            seen.add(p);
            if (n != null) {
                Iterator<Procedure> iter = n.getCallees().iterator();
                while (iter.hasNext()) {
                    Procedure o = iter.next();
                    if (!seen.contains(o))
                        horizon.add(o);
                }
            }
        }
        return false;
    }

    /**
    * Prints the graph to a stream in
    * <a href="http://www.research.att.com/sw/tools/graphviz/">graphviz</a>
    * format.
    *
    * @param stream The stream on which to print the graph.
    */
    public void print(OutputStream stream) {
        PrintStream p = new PrintStream(stream);
        p.println("digraph {\norientation=landscape;\nsize=\"11,8\";\n");
        Iterator<Procedure> l_iter = callgraph.keySet().iterator();
        while (l_iter.hasNext()) {
            Procedure l_proc = l_iter.next();
            Node node = callgraph.get(l_proc);
            p.print(l_proc.getName().toString());
            p.print(" -> { ");
            Iterator<Procedure> iter = node.getCallees().iterator();
            while (iter.hasNext()) {
                Procedure callee = iter.next();
                p.print(callee.getName().toString() + " ");
            }
            p.print("};\n");
        }
        p.print("}\n");
    }

    private void topologicalSort(Procedure proc,
                                 Set<Procedure> unvisited,
                                 List<Procedure> sorted_list) {
        Node node = callgraph.get(proc);
        unvisited.remove(proc);
        for (Procedure callee : (ArrayList<Procedure >)node.getCallees()) {
            if (unvisited.contains(callee)) {
                topologicalSort(callee, unvisited, sorted_list);
            }
        }
        sorted_list.add(proc);
    }

    public List<Procedure> getTopologicalCallList() {
        Set<Procedure> unvisited = new HashSet<Procedure>();
        unvisited.addAll(callgraph.keySet());
        ArrayList<Procedure> sorted_list =
                new ArrayList<Procedure>(unvisited.size());
        unvisited.remove(root.getProcedure());
        topologicalSort(root.getProcedure(), unvisited, sorted_list);
        return sorted_list;
    }

}
