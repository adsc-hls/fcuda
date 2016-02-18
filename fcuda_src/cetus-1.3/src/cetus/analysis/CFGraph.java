package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
* CFGraph supports creation of statmenet-level control flow graphs.
* The {@link DFANode} class is used to represent each node in the graph.
* The nodes in a CFGraph object contain information gathered from
* their corresponding Cetus IR and these data can be accessed with the
* {@link DFANode#getData(String)} method.
* Following key:data pairs are added to each node by default after calling
* the constructor of CFGraph.
*
* <ul>
* <li> ir: the IR associated with the node (either Expression or Statement).
* <li> stmt: the Statement IR whose entry node is this node.
* <li> stmt-exit: the Statement IR whose exit node is this node.
* <li> symbol-entry: symbol table object whose entry node is this node.
* <li> symbol-exit: list of symbol table object whose exit node is this node.
* <li> true: the successor node for the "taken" branch.
* <li> false: the successor node for the "not-taken" branch.
* <li> tag: additional information about each node.
* </ul>
*
* Notice that not every content above is available at a node since some nodes
* merely represents loop controls or entry/exit point of certain Cetus IRs.
*/
public class CFGraph extends DFAGraph {

    // Data structure for building CFG.
    protected Stack<List<DFANode>> break_link;
    protected Stack<List<DFANode>> continue_link;
    protected Stack<List<DFANode>> switch_link;
    // The last element of a value List<DFANode> is used as the target node
    // (label) whereas other elements are source nodes (goto).
    protected Map<Label, List<DFANode>> goto_link;

    // Class that is marked as super nodes
    protected Class<? extends Traversable> super_node;

    // Object that a CFG is created from (top-level call).
    protected Traversable root_node;

    // Include declaration node 
    protected boolean includeDeclarator = false;

    /**
    * Constructs a CFGraph object with the given traversable object.
    * The entry node contains a string "ENTRY" for the key "stmt".
    *
    * @param t the traversable object.
    */
    public CFGraph(Traversable t) {
        this(t, null);
    }

    /**
    * Constructs a CFGraph object with the given traversable object and the
    * IR type whose sub graph is pruned. The resulting control
    * flow graph does contain the sub graphs for the specified IR type but those
    * sub graphs are not connected to/from the whole graph. Depending on the
    * applications of CFGraph, those isolated sub graphs can be removed from or
    * reconnected to the whole graph.
    *
    * @param t the traversable object.
    * @param supernode IR type that is pruned.
    */
    public CFGraph(Traversable t, Class<? extends Traversable> supernode) {
        super();
        // Create space to keep track of jumps.
        break_link = new Stack<List<DFANode>>();
        continue_link = new Stack<List<DFANode>>();
        switch_link = new Stack<List<DFANode>>();
        goto_link = new HashMap<Label, List<DFANode>>();
        // Super node type.
        super_node = supernode;
        root_node = t;
        // ENTRY insertion.
        DFANode entry = new DFANode("stmt", "ENTRY");
        entry.putData("tag", "FLOW ENTRY");
        addNode(entry);
        // Build and absorb.
        DFAGraph g = buildGraph(t);
        addEdge(entry, g.getFirst());
        absorb(g);
        // Optimize.
        reduce();
    }

    // TODO: needs documentation
    @SuppressWarnings("unchecked")
    public CFGraph(VariableDeclarator t, boolean arrayDecl) {
        super();
        // Create space to keep track of jumps.
        break_link = new Stack<List<DFANode>>();
        continue_link = new Stack<List<DFANode>>();
        switch_link = new Stack<List<DFANode>>();
        goto_link = new HashMap<Label, List<DFANode>>();
        // Super node type.
        super_node = null;
        root_node = t;
        // ENTRY insertion.
        DFANode entry = new DFANode("stmt", "ENTRY");
        entry.putData("tag", "FLOW ENTRY");
        addNode(entry);
        DFAGraph ret = new DFAGraph();
        DepthFirstIterator decIter = new DepthFirstIterator(t);
        decIter.pruneOn(AssignmentExpression.class);
        decIter.pruneOn(FunctionCall.class);
        decIter.pruneOn(UnaryExpression.class);
        while (decIter.hasNext()) {
            Object o = decIter.next();
            if (o instanceof AssignmentExpression) {
                DFANode last = null;
                if (ret.size() != 0) {
                    last = ret.getLast();
                }
                DFANode node = new DFANode("stmt", o);
                node.putData("ir", o);
                ret.addNode(node);
                if (ret.size() != 1) {
                    addEdge(last, node);
                }
            } else if (o instanceof FunctionCall) {
                DFANode last = null;
                if (ret.size() != 0) {
                    last = ret.getLast();
                }
                DFANode node = new DFANode("stmt", o);
                node.putData("ir", o);
                ret.addNode(node);
                if (ret.size() != 1) {
                    addEdge(last, node);
                }
            } else if (o instanceof UnaryExpression) {
                DFANode last = null;
                if (ret.size() != 0) {
                    last = ret.getLast();
                }
                DFANode node = new DFANode("stmt", o);
                node.putData("ir", o);
                ret.addNode(node);
                if (ret.size() != 1) {
                    addEdge(last, node);
                }
            } else if (o instanceof Initializer) {
                // Other modules will handle this case.
            } else if (o instanceof VariableDeclarator) {
                // Skip itself
            } else if (o instanceof NameID) {
                // Do nothing for NameID
            } else {
                throw new
                    RuntimeException("Unexpected Class on building CFG: " +
                                     o.getClass().getCanonicalName());
            }
        }
        // Optimize.
        reduce();
    }

    // TODO: needs documentation
    @SuppressWarnings("unchecked")
    public CFGraph(NestedDeclarator t, boolean arrayDecl) {
        super();
        // Create space to keep track of jumps.
        break_link = new Stack<List<DFANode>>();
        continue_link = new Stack<List<DFANode>>();
        switch_link = new Stack<List<DFANode>>();
        goto_link = new HashMap<Label, List<DFANode>>();
        // Super node type.
        super_node = null;
        root_node = t;
        // ENTRY insertion.
        DFANode entry = new DFANode("stmt", "ENTRY");
        entry.putData("tag", "FLOW ENTRY");
        addNode(entry);
        DFAGraph ret = new DFAGraph();
        DepthFirstIterator decIter = new DepthFirstIterator(t);
        decIter.pruneOn(AssignmentExpression.class);
        decIter.pruneOn(FunctionCall.class);
        decIter.pruneOn(UnaryExpression.class);
        while (decIter.hasNext()) {
            Object o = decIter.next();
            if (o instanceof AssignmentExpression) {
                DFANode last = null;
                if (ret.size() != 0) {
                    last = ret.getLast();
                }
                DFANode node = new DFANode("stmt", o);
                node.putData("ir", o);
                ret.addNode(node);
                if (ret.size() != 1) {
                    addEdge(last, node);
                }
            } else if (o instanceof FunctionCall) {
                DFANode last = null;
                if (ret.size() != 0) {
                    last = ret.getLast();
                }
                DFANode node = new DFANode("stmt", o);
                node.putData("ir", o);
                ret.addNode(node);
                if (ret.size() != 1) {
                    addEdge(last, node);
                }
            } else if (o instanceof UnaryExpression) {
                DFANode last = null;
                if (ret.size() != 0) {
                    last = ret.getLast();
                }
                DFANode node = new DFANode("stmt", o);
                node.putData("ir", o);
                ret.addNode(node);
                if (ret.size() != 1) {
                    addEdge(last, node);
                }
            } else if (o instanceof Initializer) {
                // Other modules will handle this case.
            } else if (o instanceof NestedDeclarator) {
                // Skip itself
            } else if (o instanceof NameID) {
                // Do nothing for NameID
            } else {
                throw new
                    RuntimeException("Unexpected Class on building CFG: " +
                                     o.getClass().getCanonicalName());
            }
        }
        // Optimize.
        reduce();
    }

    public CFGraph(Traversable t, Class<? extends Traversable> supernode,
            boolean includeDeclarator, List<VariableDeclaration> paramList) {
        super();
        this.includeDeclarator = includeDeclarator;
        // Create space to keep track of jumps.
        break_link = new Stack<List<DFANode>>();
        continue_link = new Stack<List<DFANode>>();
        switch_link = new Stack<List<DFANode>>();
        goto_link = new HashMap<Label, List<DFANode>>();
        // Super node type.
        super_node = supernode;
        root_node = t;
        // ENTRY insertion.
        DFANode entry = new DFANode("stmt", "ENTRY");
        entry.putData("tag", "FLOW ENTRY");
        addNode(entry);
        for (VariableDeclaration vd : paramList) {
            if (vd.getDeclarator(0) instanceof NestedDeclarator) {
                NestedDeclarator declarator = (NestedDeclarator) vd.getDeclarator(0);
                DFANode node = new DFANode("stmt", declarator);
                node.putData("ir", declarator);
                node.putData("param", vd);
                node.putData("param_idx", paramList.indexOf(vd));
                DFANode last = getLast();
                addNode(node);
                addEdge(last, node);
            } else {
                VariableDeclarator declarator = (VariableDeclarator) vd.getDeclarator(0);
                DFANode node = new DFANode("stmt", declarator);
                node.putData("ir", declarator);
                node.putData("param", vd);
                node.putData("param_idx", paramList.indexOf(vd));
                DFANode last = getLast();
                addNode(node);
                addEdge(last, node);
            }
        }
        DFANode last = getLast();
        // Build and absorb.
        DFAGraph g = buildGraph(t);
        addEdge(last, g.getFirst());
        absorb(g);
        // Optimize.
        reduce();
    }

    /**
    * Converts the graph to a string in dot format to be used with GraphViz. By
    * default, it prints out the corresponing IR objects or tags.
    * See {@link DFAGraph#toDot(String,int)} for more flexible formatting.
    *
    * @return the string in dot format.
    * @see DFAGraph#toDot(String,int)
    */
    public String toDot() {
        return toDot("tag,ir", 1);
    }


    /**
    * Returns the corresponding IR object for the specified node.
    *
    * @param node the node in the graph.
    * @return the corresponding IR, null if no such mapping exists.
    */
    public static Object getIR(DFANode node) {
        return node.getData("ir");
    }

    public DFANode getEntry() {
        return getNodeWith("stmt", "ENTRY");
    }

    /**
    * Normalizes the graph so that each node does not have an assignment
    * expression as a sub expression of another expression. Expressions that
    * contain conditionally evaluated sub expressions -- e.g., short-circuit
    * evaluation -- are not normalized to avoid possibly unsafe expression
    * evaluation.
    */
    public void normalize() {
        if (size() < 1) {
            return;
        }
        DFANode last = getLast(), node = null;
        int i = 0;
        do {
            node = getNode(i++);
            Object o = getIR(node);
            if (o instanceof Traversable &&
                IRTools.containsClass((Traversable)o, FunctionCall.class)) {
                continue;
            }
            DFAGraph temp = expandExpression(node);
            if (temp == null) {
                continue;
            }
            // Propagates properties from the unexpanded node.
            Object data = node.getData("stmt");
            if (data != null) {
                temp.getFirst().putData("stmt", data);
                // Mark the reference to the original statement
                for (int j = 0; j < temp.nodes.size(); j++) {
                    temp.nodes.get(j).putData("stmt-ref", data);
                }
            }
            if ((data = node.getData("loop-variants")) != null) {
                temp.getFirst().putData("loop-variants", data);
            }
            if ((data = node.getData("symbol-entry")) != null) {
                temp.getFirst().putData("symbol-entry", data);
            }
            if ((data = node.getData("symbol-exit")) != null) {
                List<SymbolTable> node_symbol_exit =
                        node.getData("symbol-exit");
                List<SymbolTable> symbol_exit =
                        temp.getLast().getData("symbol-exit");
                if (symbol_exit == null) {
                    symbol_exit = new ArrayList<SymbolTable>(4);
                    temp.getLast().putData("symbol-exit", symbol_exit);
                }
                for (int j = 0; j < node_symbol_exit.size(); j++) {
                    symbol_exit.add(node_symbol_exit.get(j));
                }
            }
            // Reconnect edges
            for (DFANode pred : node.getPreds()) {
                if (pred.getData("true") == node) {
                    pred.putData("true", temp.getFirst());
                } else if (pred.getData("false") == node) {
                    pred.putData("false", temp.getFirst());
                }
                addEdge(pred, temp.getFirst());
            }
            // Reconnect edges
            for (DFANode succ : node.getSuccs()) {
                if (node.getData("true") == succ) {
                    temp.getLast().putData("true", succ);
                } else if (node.getData("false") == succ) {
                    temp.getLast().putData("false", succ);
                }
                addEdge(temp.getLast(), succ);
            }
            absorb(temp);
            removeNode(node);
            --i;
        } while (last != node);
    }

    /**
    * Builds a control flow graph for a traversable object.
    */
    protected DFAGraph buildGraph(Traversable t) {
        DFAGraph ret;
        if (t instanceof Procedure) {
            ret = buildProcedure((Procedure)t);
        } else if (t instanceof CompoundStatement) {
            ret = buildCompound((CompoundStatement)t);
        } else if (t instanceof AnnotationStatement) {
            ret = buildAnnotationStatement((AnnotationStatement)t);
        } else if (t instanceof DeclarationStatement) {
            ret = buildDeclarationStatement((DeclarationStatement)t);
        } else if (t instanceof DoLoop) {
            ret = buildDoLoop((DoLoop)t);
        } else if (t instanceof ForLoop) {
            ret = buildForLoop((ForLoop)t);
        } else if (t instanceof IfStatement) {
            ret = buildIf((IfStatement)t);
        } else if (t instanceof SwitchStatement) {
            ret = buildSwitch((SwitchStatement)t);
        } else if (t instanceof WhileLoop) {
            ret = buildWhile((WhileLoop)t);
        } else if (t instanceof BreakStatement) {
            ret = new DFAGraph();
            ret.addNode(buildBreak((Statement)t));
        } else if (t instanceof Case || t instanceof Default) {
            ret = new DFAGraph();
            ret.addNode(buildCase((Statement)t));
        } else if (t instanceof ContinueStatement) {
            ret = new DFAGraph();
            ret.addNode(buildContinue((Statement)t));
        } else if (t instanceof GotoStatement) {
            ret = new DFAGraph();
            ret.addNode(buildGoto((GotoStatement)t));
        } else if (t instanceof Label) {
            ret = new DFAGraph();
            ret.addNode(buildLabel((Label)t));
        } else {
            DFANode node = new DFANode("stmt", t);
            node.putData("ir", t);
            ret = new DFAGraph();
            ret.addNode(node);
        }
        if (t != null && t != root_node && super_node != null &&
            super_node.isInstance(t)) {
            DFANode entry = new DFANode("super-entry", t);
            entry.putData("tag", t.getClass().getName());
            DFANode exit = new DFANode("super-exit", t);
            ret = new DFAGraph();
            ret.nodes.add(0, entry);
            ret.nodes.add(exit);
            ret.addEdge(entry, exit);
        }
        return ret;
    }

    /** Check if the node contains unconditional jump instruction. */
    protected static boolean isJump(DFANode node) {
        Object ir = node.getData("ir");
        return (ir != null && (ir instanceof BreakStatement ||
                               ir instanceof ContinueStatement ||
                               ir instanceof GotoStatement ||
                               ir instanceof ReturnStatement));
    }

    // Build a graph for a Procedure.
    protected DFAGraph buildProcedure(Procedure proc) {
        DFAGraph ret = buildGraph(proc.getBody());
        // Finalizes delayed edges for goto -> label
        for (Label l : goto_link.keySet()) {
            List<DFANode> node_list = goto_link.get(l);
            DFANode to = node_list.get(node_list.size() - 1);
            for (int i = 0; i < node_list.size() - 1; i++) {
                ret.addEdge(node_list.get(i), to);
            }
        }
        return ret;
    }

    protected DFAGraph buildAnnotationStatement(AnnotationStatement stmt) {
        return new DFAGraph();
    }

    protected DFAGraph buildDeclarationStatement(DeclarationStatement stmt) {
        if (includeDeclarator) {
            DFAGraph ret = new DFAGraph();
            DFIterator<VariableDeclarator> decIter =
                    new DFIterator<VariableDeclarator>(stmt,
                    VariableDeclarator.class);
            decIter.pruneOn(VariableDeclarator.class);
            while (decIter.hasNext()) {
                VariableDeclarator o = decIter.next();
                DFANode lastNode = null;
                if (ret.size() > 0) {
                    lastNode = ret.getLast();
                }
                DFANode node = new DFANode("stmt", o);
                node.putData("ir", o);
                ret.addNode(node);
                if (lastNode != null) {
                    addEdge(lastNode, node);
                }
            }
            return ret;
        } else {
            return new DFAGraph();
        }
    }

    // Build a node for a break statement.
    protected DFANode buildBreak(Statement stmt) {
        DFANode ret = new DFANode("stmt", stmt);
        ret.putData("ir", stmt);
        // Delay links - empty stack means this graph doesn't have the target
        // statement to jump to and this node has foreign outgoing control flow 
        // path.
        if (!break_link.empty()) {
            break_link.peek().add(ret);
        }
        return ret;
    }

    // Build a node for a case/default statement.
    protected DFANode buildCase(Statement stmt) {
        DFANode ret = new DFANode("stmt", stmt);
        ret.putData("ir", stmt);
        // Delay links - empty stack means this graph doesn't have the switch
        // statement and this node has incoming control flow path from unknown
        // node.
        if (!switch_link.empty()) {
            switch_link.peek().add(ret);
        } else {
            PrintTools.printlnStatus("[WARNING] Orphan switch case pair", 0);
        }
        return ret;
    }

    // Build a node for a continue statement.
    protected DFANode buildContinue(Statement stmt) {
        DFANode ret = new DFANode("stmt", stmt);
        ret.putData("ir", stmt);
        // Delay links - empty stack means this graph doesn't have the target
        // statement to jump to and this node has outgoing control flow 
        // path to an unknown target.
        if (!continue_link.empty()) {
            continue_link.peek().add(ret);
        }
        return ret;
    }

    // Build a node for a goto statement.
    protected DFANode buildGoto(GotoStatement stmt) {
        DFANode ret = new DFANode("stmt", stmt);
        ret.putData("ir", stmt);
        // Delay links.
        Label label = stmt.getTarget();
        if (goto_link.get(label) == null) {
            goto_link.put(label, new ArrayList<DFANode>(4));
        }
        goto_link.get(label).add(0, ret);
        // Add symbol-exit attribute; add all symbol table object until it
        // reaches the same parent with the target.
        Traversable target_table = stmt.getTarget();
        while (!(target_table instanceof SymbolTable)) {
            target_table = target_table.getParent();
        }
        Traversable t = stmt.getParent();
        List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
        while (t != target_table && t != null) {
            if (t instanceof SymbolTable &&
                !((SymbolTable) t).getDeclarations().isEmpty()) {
                symbol_exits.add((SymbolTable)t);
            }
            t = t.getParent();
        }
        ret.putData("symbol-exit", symbol_exits);
        return ret;
    }

    // Build a node for a label.
    protected DFANode buildLabel(Label label) {
        DFANode ret = new DFANode("stmt", label);
        ret.putData("ir", label);
        // Delay links.
        if (goto_link.get(label) == null) {
            goto_link.put(label, new ArrayList<DFANode>(4));
        }
        goto_link.get(label).add(ret);
        return ret;
    }

    // Build a graph for a compound statement.
    protected DFAGraph buildCompound(CompoundStatement stmt) {
        DFAGraph ret = new DFAGraph();
        // Absorbs subgraph from each child.
        List<Traversable> children = stmt.getChildren();
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            DFAGraph curr = buildGraph(children.get(i));
            // Jumps are not connected to the next statement.
            if (!ret.isEmpty() && !curr.isEmpty() && !isJump(ret.getLast())) {
                ret.addEdge(ret.getLast(), curr.getFirst());
            }
            ret.absorb(curr);
        }
        // Insert an empty node if this compound statement has no children.
        if (ret.size() == 0) {
            ret.addNode(new DFANode("ir", new NullStatement()));
        }
        // Record live period of symbols by adding pointer to the symbol table
        if (!stmt.getDeclarations().isEmpty()) {
            ret.getFirst().putData("symbol-entry", stmt);
            List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
            symbol_exits.add(stmt);
            ret.getLast().putData("symbol-exit", symbol_exits);
        }
        return ret;
    }

    // Build a graph for a do while loop.
    protected DFAGraph buildDoLoop(DoLoop stmt) {
        DFAGraph ret = new DFAGraph();
        CompoundStatement bs = (CompoundStatement)stmt.getBody();
        // Build nodes.
        DFANode entry = new DFANode("stmt", stmt);
        DFANode condition = new DFANode("ir", stmt.getCondition());
        DFANode exit = new DFANode("stmt-exit", stmt);
        // Delay links.
        break_link.push(new ArrayList<DFANode>(4));
        continue_link.push(new ArrayList<DFANode>(4));
        // Build subgraph.
        DFAGraph body = buildGraph(bs);
        // Put data.
        entry.putData("tag", "DOLOOP");
        entry.putData("do-condition", condition);
        entry.putData("do-exit", exit);
        condition.putData("true", body.getFirst());
        condition.putData("false", exit);
        condition.putData("loop-variants", DataFlowTools.getDefSymbol(stmt));
        if (!bs.getDeclarations().isEmpty()) {
            List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
            symbol_exits.add(bs);
            exit.putData("symbol-exit", symbol_exits);
        }
        // Add edges; entry = ret[0] and exit = ret[last]
        ret.addEdge(entry, body.getFirst());
        ret.absorb(body);
        if (!isJump(body.getLast())) {
            ret.addEdge(body.getLast(), condition);
        }
        ret.addEdge(condition, body.getFirst());
        ret.addEdge(condition, exit);
        // Finalize delayed jumps.
        List<DFANode> break_links = break_link.pop();
        for (int i = 0; i < break_links.size(); i++) {
            ret.addEdge(break_links.get(i), exit);
        }
        List<DFANode> continue_links = continue_link.pop();
        for (int i = 0; i < continue_links.size(); i++) {
            ret.addEdge(continue_links.get(i), condition);
        }
        return ret;
    }

    // Build a graph for a for loop.
    protected DFAGraph buildForLoop(ForLoop stmt) {
        DFAGraph ret = new DFAGraph();
        CompoundStatement bs = (CompoundStatement)stmt.getBody();
        // Build nodes.
        DFANode init = new DFANode("stmt", stmt);
        DFANode condition = new DFANode("ir", stmt.getCondition());
        DFANode step = new DFANode("ir", stmt.getStep());
        DFANode exit = new DFANode("stmt-exit", stmt);
        // Delay links.
        break_link.push(new ArrayList<DFANode>(4));
        continue_link.push(new ArrayList<DFANode>(4));
        // Build subgraph.
        DFAGraph body = buildGraph(bs);
        // Put data.
        init.putData("ir", stmt.getInitialStatement());
        init.putData("for-condition", condition);
        init.putData("for-step", step);
        init.putData("for-exit", exit);
        // Keep special string for null condition (should be a unique entity).
        if (stmt.getCondition() == null) {
            condition.putData("ir", new NullStatement());
        }
        condition.putData("true", body.getFirst());
        condition.putData("false", exit);
        // Add loop variants
        condition.putData("loop-variants", DataFlowTools.getDefSymbol(stmt));
        if (!bs.getDeclarations().isEmpty()) {
            List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
            symbol_exits.add(bs);
            exit.putData("symbol-exit", symbol_exits);
        }
        // Keep special string for null step (should be a unique entity).
        if (stmt.getStep() == null) {
            step.putData("ir", new NullStatement());
        }
        exit.putData("tag", "FOREXIT");
        // Add edges; init = ret[0] and exit = ret[last].
        ret.addEdge(init, condition);
        ret.addEdge(condition, body.getFirst());
        ret.absorb(body);
        if (!isJump(body.getLast())) {
            ret.addEdge(body.getLast(), step);
        }
        ret.addEdge(step, condition);
        ret.addEdge(condition, exit);
        // Finalize delayed jumps.
        List<DFANode> break_links = break_link.pop();
        for (int i = 0; i < break_links.size(); i++) {
            ret.addEdge(break_links.get(i), exit);
        }
        List<DFANode> continue_links = continue_link.pop();
        for (int i = 0; i < continue_links.size(); i++) {
            ret.addEdge(continue_links.get(i), step);
        }
        return ret;
    }

    // Build a graph for an if statement.
    protected DFAGraph buildIf(IfStatement stmt) {
        DFAGraph ret = new DFAGraph();
        // Build nodes
        DFANode entry = new DFANode("stmt", stmt);
        DFANode exit = new DFANode();
        DFAGraph thenG = buildGraph(stmt.getThenStatement());
        DFAGraph elseG = buildGraph(stmt.getElseStatement());
        // Put data
        entry.putData("ir", stmt.getControlExpression());
        entry.putData("true", thenG.getFirst());
        entry.putData("false", elseG.getFirst());
        exit.putData("tag", "IFEXIT");
        // Add edges; the entry/exit nodes should be ret[0] and ret[last].
        ret.addEdge(entry, thenG.getFirst());
        ret.addEdge(entry, elseG.getFirst());
        ret.absorb(thenG);
        ret.absorb(elseG);
        if (!isJump(thenG.getLast())) {
            ret.addEdge(thenG.getLast(), exit);
        }
        if (!isJump(elseG.getLast())) {
            ret.addEdge(elseG.getLast(), exit);
        }
        return ret;
    }

    // Build a graph for a switch statement.
    protected DFAGraph buildSwitch(SwitchStatement stmt) {
        DFAGraph ret = new DFAGraph();
        CompoundStatement bs = stmt.getBody();
        // Build nodes.
        DFANode entry = new DFANode("stmt", stmt);
        DFANode exit = new DFANode("stmt-exit", stmt);
        // Delay links.
        break_link.push(new ArrayList<DFANode>(4));
        switch_link.push(new ArrayList<DFANode>(4));
        // Build subgraph.
        DFAGraph body = buildGraph(bs);
        // Put data.
        entry.putData("ir", stmt);
        entry.putData("switch-exit", exit);
        entry.putData("tag", "switch(" + stmt.getExpression() + ")");
        exit.putData("tag", "SWITCHEXIT");
        if (!bs.getDeclarations().isEmpty()) {
            List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
            symbol_exits.add(bs);
            exit.putData("symbol-exit", symbol_exits);
        }
        // Add edges; entry = ret[0] and exit = ret[last].
        ret.addNode(entry);     // jumps are delayed
        ret.absorb(body);
        ret.addNode(exit);
        if (!isJump(body.getLast())) {
            ret.addEdge(body.getLast(), exit);
        }
        // Finalize delayed jumps.
        List<DFANode> break_links = break_link.pop();
        for (int i = 0; i < break_links.size(); i++) {
            ret.addEdge(break_links.get(i), exit);
        }
        List<DFANode> switch_links = switch_link.pop();
        for (int i = 0; i < switch_links.size(); i++) {
            ret.addEdge(entry, switch_links.get(i));
        }
        return ret;
    }

    // Build a graph for a while statement.
    protected DFAGraph buildWhile(WhileLoop stmt) {
        DFAGraph ret = new DFAGraph();
        CompoundStatement bs = (CompoundStatement)stmt.getBody();
        // Build nodes.
        DFANode entry = new DFANode("stmt", stmt);
        DFANode condition = new DFANode("ir", stmt.getCondition());
        DFANode exit = new DFANode("stmt-exit", stmt);
        // Delay links.
        break_link.push(new ArrayList<DFANode>(4));
        continue_link.push(new ArrayList<DFANode>(4));
        // Build subgraph.
        DFAGraph body = buildGraph(bs);
        // Put data.
        entry.putData("tag", "WHILE");
        entry.putData("while-condition", condition);
        entry.putData("while-exit", exit);
        condition.putData("true", body.getFirst());
        condition.putData("false", exit);
        condition.putData("loop-variants", DataFlowTools.getDefSymbol(stmt));
        exit.putData("tag", "WHILEEXIT");
        if (!bs.getDeclarations().isEmpty()) {
            List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
            symbol_exits.add(bs);
            exit.putData("symbol-exit", symbol_exits);
        }
        // Add edges; entry = ret[0] and exit = ret[last].
        ret.addEdge(entry, condition);
        ret.absorb(body);
        ret.addEdge(condition, body.getFirst());
        ret.addEdge(condition, exit);
        if (!isJump(body.getLast())) {
            ret.addEdge(body.getLast(), condition);
        }
        // Finalize delayed jumps.
        List<DFANode> break_links = break_link.pop();
        for (int i = 0; i < break_links.size(); i++) {
            ret.addEdge(break_links.get(i), exit);
        }
        List<DFANode> continue_links = continue_link.pop();
        for (int i = 0; i < continue_links.size(); i++) {
            ret.addEdge(continue_links.get(i), condition);
        }
        return ret;
    }

    protected boolean removable_node(DFANode node) {
        if (node.getSuccs().size() == 1 &&
            node.getData(Arrays.asList("stmt", "ir", "symbol-exit",
                    "stmt-exit", "super-entry")) == null) {
            return true;
        } else {
            return false;
        }
    }

    // Reduce graph by removing nodes with empty IR.
    protected void reduce() {
        if (size() < 1) {
            return;
        }
        DFANode last = getLast(), node = null;
        int i = 0;
        do {
            node = getNode(i++);
            // Find node with no cetus IR and with a single successor.
            if (removable_node(node)) {
                Set<DFANode> preds = node.getPreds();
                Set<DFANode> succs = node.getSuccs();
                removeNode(node);
                --i;            // Graph was resized.
                // Reconnect edges
                for (DFANode pred : preds) {
                    for (DFANode succ : succs) {
                        // Inherit edge property
                        if (pred.getData("true") == node) {
                            pred.putData("true", succ);
                        } else if (pred.getData("false") == node) {
                            pred.putData("false", succ);
                        }
                        addEdge(pred, succ);
                    }
                }
            }
        } while (last != node);
    }

    // Check if the given ir has any irreversible side effects when expanded.
    // This situation happens if the back-end compiler has short-circuit
    // evaluation ability, which is common.
    protected static boolean isUnsafeExpansion(Traversable t) {
        // Collect chunks of expressions that are evaluated at once.
        List<Traversable> chunks = getSafeEvaluation(t);
        if (chunks.size() < 2) {
            return false;
        }
        for (int i = 1; i < chunks.size(); ++i) {
            if (IRTools.containsSideEffect(chunks.get(i))) {
                return true;
            }
        }
        return false;
    }

    // Collects list of expressions in evaluation order. It is guaranteed that
    // each element of the returned list is at least evaluated in some
    // sequential order without any branches at runtime.
    protected static List<Traversable> getSafeEvaluation(Traversable t) {
        List<Traversable> ret = new ArrayList<Traversable>(4);
        if (t instanceof UnaryExpression &&
                ((UnaryExpression)t).getOperator() ==
                        UnaryOperator.LOGICAL_NEGATION) {
            ret.addAll(getSafeEvaluation(t.getChildren().get(0)));
        } else if (t instanceof BinaryExpression &&
                (((BinaryExpression)t).getOperator() ==
                        BinaryOperator.LOGICAL_AND ||
                ((BinaryExpression)t).getOperator() ==
                        BinaryOperator.LOGICAL_OR)) {
            ret.addAll(getSafeEvaluation(t.getChildren().get(0)));
            ret.addAll(getSafeEvaluation(t.getChildren().get(1)));
        } else if (ret.size() == 0) {
            ret.add(t);
        }
        return ret;
    }

    // Normalizes the IR in the node so that each statement has at most one
    // assignment expression. The original IR does not change with this method;
    // a new temporary IR is created and it resides only on the control flow
    // graph.
    protected DFAGraph expandExpression(DFANode node) {
        Object o = node.getData("ir");
        // Other types of statements does not carry side effects.
        if (!(o instanceof Expression || o instanceof ExpressionStatement)) {
            return null;
        }
        // Branches with short-circuit evaluation are not expanded.
        if (isUnsafeExpansion((Traversable)o)) {
            return null;
        }
        // Build a compound statement for this node.
        CompoundStatement stmts = new CompoundStatement();
        Traversable t = null;
        if (o instanceof Expression) {
            t = new ExpressionStatement(((Expression)o).clone());
        } else {
            t = ((Statement)o).clone();
        }
        stmts.addStatement((Statement)t);
        Class before = t.getChildren().get(0).getClass();
        expandExpression(stmts, t);
        // Remove redundant repetition of lvalue.
        Class after = t.getChildren().get(0).getClass();
        if (t.getParent() == stmts && before != after &&
                before == AssignmentExpression.class) {
            stmts.removeChild(t);
        }
        // Just use the buildGraph method for the compound statement.
        DFAGraph ret = new CFGraph(stmts);
        ret.removeNode(ret.getNode("stmt", "ENTRY"));
        // Remove "stmt" locators.
        int ret_size = ret.size();
        for (int i = 0; i < ret_size; i++) {
            DFANode tempnode = ret.nodes.get(i);
            if (tempnode.getData("stmt") != null) {
                tempnode.removeData("stmt");
            }
        }
        // Remember the exit point of temporary IR for clean-up.
        if (!stmts.getDeclarations().isEmpty()) {
            // Use list of symbol tables to save spaces
            List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
            symbol_exits.add(stmts);
            ret.getLast().putData("symbol-exit", symbol_exits);
        }
        return ret;
    }

    // Expand assignments; a+=b ==> temp=a+b; a=temp;
    protected void
            expandAssignment(CompoundStatement stmts, AssignmentExpression e) {
        AssignmentExpression ae = e.clone();
        AssignmentOperator op = ae.getOperator();
        // Normalize assignment.
        if (op != AssignmentOperator.NORMAL) {
            BinaryOperator bop =
                    BinaryOperator.fromString(op.toString().replace("=", ""));
            ae.getRHS().swapWith(new BinaryExpression(
                    ae.getLHS().clone(), bop, ae.getRHS().clone()));
            ae.setOperator(AssignmentOperator.NORMAL);
        }
        // Add the normalized assignment.
        Object last = stmts.getChildren().get(stmts.getChildren().size() - 1);
        Statement stmt = new ExpressionStatement(ae);
        stmts.addStatementBefore((Statement) last, stmt);
        // Replace the original assignment with lhs.
        e.swapWith(e.getLHS());
    }

    // Expand conditional expressions; assign temporary lvalue.
    protected void
            expandConditional(CompoundStatement stmts, ConditionalExpression e){
        ConditionalExpression ce = e.clone();
        // Just consider they are integer type for now; further analysis will
        // detect if range is extractable anyhow.
        Identifier id = SymbolTools.getTemp(stmts, Specifier.INT, null);
        // Build an if statement.
        AssignmentOperator op = AssignmentOperator.NORMAL;
        Expression etrue = new AssignmentExpression(
                id, op, ce.getTrueExpression().clone());
        Expression efalse = new AssignmentExpression(
                id.clone(), op, ce.getFalseExpression().clone());
        Statement ifstmt = new IfStatement(
                ce.getCondition().clone(), new ExpressionStatement(etrue),
                new ExpressionStatement(efalse));
        // Add if statement to the list
        Object last = stmts.getChildren().get(stmts.getChildren().size() - 1);
        stmts.addStatementBefore((Statement) last, ifstmt);
        // Replacement
        e.swapWith((Expression) id.clone());
    }

    // Expand unary expressions; assign temporary lvalue.
    protected void expandUnary(CompoundStatement stmts, UnaryExpression e) {
        UnaryOperator op = e.getOperator();
        if (op != UnaryOperator.POST_DECREMENT &&
            op != UnaryOperator.POST_INCREMENT &&
            op != UnaryOperator.PRE_DECREMENT &&
            op != UnaryOperator.PRE_INCREMENT) {
            return;
        }
        // Get a temporary variable.
        Identifier id = SymbolTools.getTemp(stmts, Specifier.INT, null);
        Expression var = e.getExpression();
        // Add assignments to the list.
        AssignmentOperator aop = AssignmentOperator.NORMAL;
        Expression temp = new AssignmentExpression(id.clone(), aop,var.clone());
        Expression rhs = null;
        if (op == UnaryOperator.POST_DECREMENT ||
            op == UnaryOperator.PRE_DECREMENT) {
            rhs = new BinaryExpression(
                    var.clone(), BinaryOperator.SUBTRACT,new IntegerLiteral(1));
        } else {
            rhs = new BinaryExpression(
                    var.clone(), BinaryOperator.ADD, new IntegerLiteral(1));
        }
        Expression diff = new AssignmentExpression(var.clone(), aop, rhs);
        Object last = stmts.getChildren().get(stmts.getChildren().size() - 1);
        if (op == UnaryOperator.POST_DECREMENT ||
            op == UnaryOperator.POST_INCREMENT) {
            stmts.addStatementBefore(
                    (Statement)last, new ExpressionStatement(temp));
            stmts.addStatementBefore(
                    (Statement)last, new ExpressionStatement(diff));
        } else {
            stmts.addStatementBefore(
                    (Statement)last, new ExpressionStatement(diff));
            stmts.addStatementBefore(
                    (Statement) last, new ExpressionStatement(temp));
        }
        // Replacement.
        e.swapWith(id.clone());
    }

    // Remove redundant IR created from normalization; this method performs a
    // simple live variable analysis for the temporary variables, and remove
    // redundant write operation to the temproary variables.
    protected static void removeRedundancy(CompoundStatement stmts) {
        Set<Symbol> lives = new HashSet<Symbol>();
        Set<Symbol> kills = new HashSet<Symbol>();
        List stmt_list = stmts.getChildren();
        for (int i = stmt_list.size() - 1; i >= 0; --i) {
            Statement stmt = (Statement)stmt_list.get(i);
            // Skip declarations
            if (!(stmt instanceof ExpressionStatement)) {
                continue;
            }
            Expression e = ((ExpressionStatement)stmt).getExpression();
            // Identifiers
            if (e instanceof Identifier) {
                stmts.removeChild(stmt);
                continue;
            // Comma identifiers
            } else if (e instanceof CommaExpression) {
                boolean is_comma_ids = true;
                List<Traversable> children = e.getChildren();
                for (int j = 0; j < children.size(); j++) {
                    if (!(children.get(j) instanceof Identifier)) {
                        is_comma_ids = false;
                        break;
                    }
                }
                if (is_comma_ids) {
                    stmts.removeChild(stmt);
                    continue;
                }
            // Temporary assignments with no live use; its declaration is also
            // deleted afterwards.
            } else if (e instanceof AssignmentExpression) {
                Expression to = ((AssignmentExpression)e).getLHS();
                if (to instanceof Identifier) {
                    Symbol var = ((Identifier) to).getSymbol();
                    if (SymbolTools.getSymbols(stmts).contains(var) &&
                            !lives.contains(var)) {
                        stmts.removeChild(stmt);
                        kills.add(var);
                        continue;
                    }
                }
            }
            // Record live temp variables
            for (Symbol var : SymbolTools.getAccessedSymbols(e)) {
                if (stmts.containsSymbol(var)) {
                    lives.add(var);
                }
            }
        }
        // Additional round that cleans up unused temporary declarations.
        //System.out.println("Before: "+SymbolTools.getSymbols(stmts));
        List<Traversable> children = stmts.getChildren();
        for (int i = 0; i < children.size(); i++) {
            if (IRTools.containsSymbols(children.get(i), kills)) {
                children.remove(i--);
            }
        }
        //System.out.println("After: "+SymbolTools.getSymbols(stmts));
    }

    // Expand expression at top level.
    protected void expandExpression(CompoundStatement stmts, Traversable t) {
        List<Traversable> children = t.getChildren();
        for (int i = 0; i < children.size(); i++) {
            expandExpression(stmts, children.get(i));
        }
        if (t instanceof AssignmentExpression) {
            expandAssignment(stmts, (AssignmentExpression)t);
        } else if (t instanceof ConditionalExpression) {
            expandConditional(stmts, (ConditionalExpression)t);
        } else if (t instanceof UnaryExpression) {
            expandUnary(stmts, (UnaryExpression)t);
        }
        if (stmts.getChildren().size() > 1) {   // Don't touch the original IR.
            removeRedundancy(stmts);
        }
    }

    /**
    * Returns the DFANode object that contains the given function call.
    */
    public DFANode getCallNode(FunctionCall fc) {
        int nodes_size = nodes.size();
        for (int i = 0; i < nodes_size; i++) {
            DFANode node = nodes.get(i);
            Object o = getIR(node);
            if (o instanceof Traversable) {
                DFIterator<FunctionCall> iter = new DFIterator<FunctionCall>(
                        (Traversable)o, FunctionCall.class);
                while (iter.hasNext()) {
                    if (iter.next() == fc) {
                        return node;
                    }
                }
            }
        }
        //System.out.println("fc = " + fc);
        //System.out.println(toDot("ir", 1));
        //System.out.println(root_node);
        return null;
    }
}
