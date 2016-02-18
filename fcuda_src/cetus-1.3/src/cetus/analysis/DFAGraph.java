package cetus.analysis;

import cetus.hir.PrintTools;

import java.util.*;

/**
 * Class DFAGraph represents a directed graph with a set of {@link DFANode}
 * objects. This class was designed for program analysis that works on a graph
 * representation but is also useful for implementing any graph algorithms.
 * Currently available algorithms include reverse post ordering
 * (topological ordering) and Tarjan's strongly-connected-components (SCC)
 * algorithm.
 */
public class DFAGraph {

    /** List that contains all nodes in this graph. */
    protected ArrayList<DFANode> nodes;

    /**
     * Constructs an empty DFAGraph object.
     */
    public DFAGraph() {
        nodes = new ArrayList<DFANode>();
    }

    /**
     * Adds a node to the graph if the node does not exist in the graph.
     *
     * @param node the node being added.
     */
    public void addNode(DFANode node) {
        if (!nodes.contains(node)) {
            nodes.add(node);
        }
    }

    /**
     * Absorbs all nodes from another graph.
     *
     * @param other the graph being absorbed.
     */
    public void absorb(DFAGraph other) {
        List<DFANode> other_nodes = other.nodes;
        int other_size = other_nodes.size();
        for (int i = 0; i < other_size; i++) {
            addNode(other_nodes.get(i));
        }
    }

    /**
     * Returns the node indexed by the specified id in the node list.
     *
     * @param id the index of the node.
     * @return the node indexed by id.
     */
    public DFANode getNode(int id) {
        return nodes.get(id);
    }

    /**
     * Returns the node that contains the specified key/data pair.
     *
     * @param key the key string.
     * @param data the data object.
     * @return the node if there exists a node such that
     *      data == node.getData(key), null if no such node exists.
     */
    public DFANode getNode(String key, Object data) {
        if (key == null || data == null) {
            return null;
        }
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            if (node.getData(key) == data) {
                return node;
            }
        }
        return null;
    }

    /**
     * Returns the node that contains the specified key/data pair.
     *
     * @param key the key string.
     * @param data the data object.
     * @return the node if there exists a node such that
     *      data.equals(node.getData(key)), null if no such node exists.
     */
    public DFANode getNodeWith(String key, Object data) {
        if (key == null || data == null) {
            return null;
        }
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            if (data.equals(node.getData(key))) {
                return node;
            }
        }
        return null;
    }

    /**
     * Returns the first node in the node list. This method is useful only when
     * constructing a {@link CFGraph} object.
     *
     * @return the first entry in the node list.
     */
    public DFANode getFirst() {
        return nodes.get(0);
    }

    /**
     * Returns the last node in the node list. This method is useful only when
     * constructing a {@link CFGraph} object.
     *
     * @return the last entry in the node list.
     */
    public DFANode getLast() {
        return nodes.get(nodes.size()-1);
    }

    /**
     * Returns the list of the nodes that have no successors.
     *
     * @return the list of the nodes.
     */
    public List<DFANode> getExitNodes() {
        List<DFANode> ret = new ArrayList<DFANode>();
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            if (node.getSuccs().isEmpty()) {
                ret.add(node);
            }
        }
        return ret;
    }

    /**
     * Returns the list of the nodes that have no predecessors.
     * @return the list of such nodes.
     */
    public List<DFANode> getEntryNodes() {
        List<DFANode> ret = new ArrayList<DFANode>();
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            if (node.getPreds().isEmpty()) {
                ret.add(node);
            }
        }
        return ret;
    }

    /**
     * Checks if the graph is empty.
     *
     * @return true if it is, false otherwise.
     */
    public boolean isEmpty() {
        return nodes.isEmpty();
    }

    /**
     * Returns the number of nodes contained in this graph.
     *
     * @return the size.
     */
    public int size() {
        return nodes.size();
    }

    /**
     * Adds a directed edge from a node to the other. Nodes are also added to
     * the graph if they are not present in the graph.
     *
     * @param from the source node.
     * @param to the target node.
     */
    public void addEdge(DFANode from, DFANode to) {
        addNode(from);
        from.addSucc(to);
        addNode(to);
        to.addPred(from);
    }

    /**
     * Removes a node and its associated edges from the graph.
     *
     * @param node the node being removed.
     */
    public void removeNode(DFANode node) {
        if (!nodes.contains(node)) {
            return;
        }
        for (DFANode pred : node.getPreds()) {
            pred.removeSucc(node);
        }
        for (DFANode succ : node.getSuccs()) {
            succ.removePred(node);
        }
        nodes.remove(node);
    }

    /**
     * Removes the specified nodes and their associated edges from the graph.
     *
     * @param nodes the list of nodes to be removed.
     */
    public void removeNodes(List<DFANode> nodes) {
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            removeNode(nodes.get(i));
        }
    }

    /**
     * Removes an edge from the graph.
     *
     * @param from the source node.
     * @param to the target node.
     */
    public void removeEdge(DFANode from, DFANode to) {
        from.removeSucc(to);
        to.removePred(from);
    }

    /**
     * Returns a string that shows the contents of the graph (debugging).
     *
     * @return the contents of the graph in string.
     */
    @Override
    public String toString() {
        return "<DFAGraph>\n" + nodes + "</DFAGraph>\n";
    }

    /**
     * Returns a strongly-connected components (SCC) forest in the graph
     * computed by the Tarjan's algorithm.
     *
     * @param root the starting node of depth-first search.
     * @return a list of lists of nodes (SCC forest).
     */
    public List getSCC(DFANode root) {
        Map<DFANode, int[]> id = new HashMap<DFANode, int[]>();
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            id.put(nodes.get(i), new int[2]);
        }
        ArrayList ret = new ArrayList();
        Integer index = new Integer(1);
        doSCC(root, ret, index, id);
        return ret;
    }

    /**
     * Finds SCC with Tarjan's algorithm.
     */
    @SuppressWarnings("unchecked") // list is reused for different purposes.
    private void doSCC(
            DFANode node, ArrayList list, Integer index, Map<DFANode,int[]> id){
        int[] node_id = id.get(node);
        node_id[0] = index;
        node_id[1] = index++;
        list.add(0, node);
        for (DFANode succ : node.getSuccs()) {
            int[] succ_id = id.get(succ);
            if (succ_id[0] < 1) {
                doSCC(succ, list, index, id);
                if (succ_id[1] < node_id[1]) {
                    node_id[1] = succ_id[1];
                }
            } else if (list.contains(succ)) {
                if (succ_id[0] < node_id[1]) {
                    node_id[1] = succ_id[0];
                }
            }
        }
        if (node_id[0] == node_id[1]) {
            ArrayList tree = new ArrayList();
            int to = list.indexOf(node);
            tree.addAll(list.subList(0, to+1));
            list.subList(0, to+1).clear();
            list.add(tree);
        }
    }

    /**
     * Records the reverse post (topological) order of each node; this is
     * basically * DFS with reverse post numbering.
     */
    private void topSort(DFANode node, int order[]) {
        node.putData("dfs-visited", new Boolean(true));
        for (DFANode succ : node.getSuccs()) {
            if (succ.getData("dfs-visited") == null) {
                topSort(succ, order);
            }
        }
        node.putData("top-order", new Integer(order[0]--));
    }

    private void topSortLoopCentric(DFANode node, int order[]) {
        node.putData("dfs-visited", new Boolean(true));
        LinkedList<DFANode> succ_priority_order = new LinkedList<DFANode>();
        for (DFANode succ : node.getSuccs()) {
            Object tag = succ.getData("tag");
            if ( tag != null &&
                    (tag.equals("FOREXIT") || tag.equals("WHILEEXIT"))) {
                succ_priority_order.addFirst(succ);
            } else {
                succ_priority_order.addLast(succ);
            }
        }
        for (DFANode succ : succ_priority_order) {
            if (succ.getData("dfs-visited") == null) {
                topSortLoopCentric(succ, order);
            }
        }
        node.putData("top-order", new Integer(order[0]--));    
    }
        
    /**
     * Computes and records the topological ordering of each node starting from
     * the root node. After calling this method, each node contains an integer
     * number (order) mapped by the key "top-order" ranging from
     * #total_nodes-#reachable_nodes to #total_nodes-1 and the "top-order" of an
     * unreachable node is -1. For example, the nodes of a graph with no
     * unreachable nodes are numbered from 0 to #total_nodes-1.
     *
     * @param root the starting node of depth-first search.
     * @return the least non-negative numbering of all nodes in the graph.
     */
    public int topologicalSort(DFANode root) {
        int order[] = { nodes.size()-1 }; // Pass int by reference
        topSort(root, order);
        // Clean up any temporary data fields.
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            if (node.getData("dfs-visited") == null) {
                node.putData("top-order", new Integer(-1));
                // -1 for unreachable nodes.
            } else {
                node.removeData("dfs-visited");
            }
        }
        return (order[0]+1);
    }

    /**
     * Uses the same reverse-post ordering algorithm as simple topological
     * sort. But in the presence of cycles introduced by loops, the 
     * ordering is optimized to ensure that the entire loop body is 
     * reverse-post ordered before the loop exit statements. This can 
     * help decrease time during an iterative data flow analysis 
     * where subsequent program statements are analyzed without having 
     * complete information from the loop. This strategy is also 
     * essential for specific analyses such as points-to analysis
     * 
     * @param root The starting node for reverse-post ordering
     * @return as with simple topological sort, returns the order of the 
     * first reverse-post ordered node
     */
    public int topologicalSortOptimized(DFANode root) {
        int order[] = { nodes.size() - 1 };
        topSortLoopCentric(root, order);
        // Clean up any temporary data fields.
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            if (node.getData("dfs-visited") == null) {
                node.putData("top-order", new Integer(-1));
                // -1 for unreachable nodes.
            } else {
                node.removeData("dfs-visited");
            }
        }
        return (order[0]+1);
    }

    /**
     * Checks if the "to" node is reachable from the "from" node by performing
     * a depth-first-search.
     * @param from the start node.
     * @param to the destination node.
     * @return true if it is reachable, false otherwise.
     */
    public boolean isReachable(DFANode from, DFANode to) {
        for (DFANode succ : from.getSuccs()) {
            if (isReachable(succ, to)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Converts the graph to a string in dot format with the given keys.
     * The nodes in the resulting dot format contain node labels mapped by the
     * given keys. For example, toDot("stmt", 1) will label each node with the
     * data object mapped by "stmt".
     *
     * @param keys the keys used in the search for the labels.
     * @param num the number of labels being printed.
     * @return the result string.
     */
    public String toDot(String keys, int num) {
        StringBuilder str = new StringBuilder();
        String sep = PrintTools.line_sep;
        str.append("digraph G {").append(sep);
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            str.append("  node").append(i).append(" ");
            str.append(node.toDot(keys, num)).append(sep);
        }
        for (int i = 0; i < nodes_size; i++) {
            DFANode pred = nodes.get(i);
            for (DFANode succ : pred.getSuccs()) {
                str.append("node").append(i).append("->node");
                str.append(nodes.indexOf(succ)).append("; ");
            }
        }
        str.append(sep).append("}").append(sep);
        return str.toString();
    }

    /**
     * Returns an iterator of the nodes.
     *
     * @return the iterator.
     */
    public Iterator<DFANode> iterator() {
        return nodes.iterator();
    }

    public class TopIterator implements Iterator<DFANode> {

        private TreeMap<Integer, DFANode> order;

        private Set<DFANode> already_visited;

        public TopIterator() {
            order = new TreeMap<Integer, DFANode>();
            already_visited = new HashSet<DFANode>();
            for (DFANode node : getEntryNodes()) {
                Integer top_order = node.getData("top-order");
                if (top_order == null) {
                    PrintTools.printlnStatus("[ERROR] no ordering exists", 0);
                    throw new InternalError();
                }
                if (top_order >= 0) {
                    order.put(top_order, node);
                }
            }
        }

        public TopIterator(DFANode root) {
            order = new TreeMap<Integer, DFANode>();
            already_visited = new HashSet<DFANode>();
            Integer top_order = root.getData("top-order");
            if (top_order == null) {
                PrintTools.printlnStatus("[ERROR] no ordering exists", 0);
                throw new InternalError();
            }
            if (top_order >= 0) {
                order.put(top_order, root);
            }
        }

        public boolean hasNext() {
            return (order.size() > 0);
        }

        public DFANode next() {
            DFANode ret = order.remove(order.firstKey());
            already_visited.add(ret);
            for (DFANode succ : ret.getSuccs()) {
                if (!already_visited.contains(succ)) {
                    order.put((Integer)succ.getData("top-order"), succ);
                }
            }
            return ret;
        }

        public void remove() {
            PrintTools.printlnStatus("[WARNING] remove is not permitted", 0);
        }
    }
}
