package cetus.analysis;

import cetus.hir.*;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.*;

/**
 * A control-flow graph of the program.
 * @deprecated
 */
@Deprecated
public class ControlFlowGraph {

    private static String pass_name = "[ControlFlowGraph]";

    BasicBlock rootBB = new BasicBlock();

    HashMap<Procedure, ProcBasicBlock> procBB =
            new HashMap<Procedure, ProcBasicBlock>();

    HashMap<Procedure, List<BasicBlock>> proccalls =
            new HashMap<Procedure, List<BasicBlock>>();

    ArrayList<BasicBlock> bblist = new ArrayList<BasicBlock>();

    ArrayList<BasicBlock> bbtoadjust = new ArrayList<BasicBlock>();

    BasicBlock entrybb, exitbb, currpred;

    Stack<BasicBlock> scopexitstack = new Stack<BasicBlock>();

    Stack<BasicBlock> scopentrystack = new Stack<BasicBlock>();

    Stack<BasicBlock> breakstack = new Stack<BasicBlock>();

    Stack<BasicBlock> continuestack = new Stack<BasicBlock>();

    Stack<BasicBlock> switchstack = new Stack<BasicBlock>();

    @SuppressWarnings("unchecked")
    public ControlFlowGraph(Program prog) {
        BreadthFirstIterator iter = new BreadthFirstIterator(prog);
        iter.pruneOn(Procedure.class);
        for (;;) {
            Procedure proc = null;
            try {
                proc = (Procedure)iter.next(Procedure.class);
            }
            catch(NoSuchElementException e) {
                break;
            }
            PrintTools.printlnStatus(pass_name +
                                     " creating CFG for procedure " +
                                     proc.getName().toString(), 2);
            ProcBasicBlock l_bb = getBBforProc(proc);
            procBB.put(proc, l_bb);
            String name = proc.getName().toString();
            if (name.equals("main") || name.equals("MAIN__")) {
                rootBB = l_bb.entryBB;
            }
        }
        optimizeBasicBlocks();
    }

    @SuppressWarnings("unchecked")
    private boolean containsFC(Statement p_stmt) {
        boolean hasFC = false;
        //Get an iterator on the Statement
        DepthFirstIterator l_dir = new DepthFirstIterator(p_stmt);
        while (l_dir.hasNext()) {
            try {
                FunctionCall fc = (FunctionCall)l_dir.next(FunctionCall.class);
                if (fc.getProcedure() != null) {
                    return true;
                }
            } catch(NoSuchElementException e) {
                hasFC = false;
            }
        }
        return hasFC;
    }

    public BasicBlock getRoot() {
        return rootBB;
    }

    /**
    * Removes redundant empty blocks.
    */
    private void optimizeBasicBlocks() {
        Iterator<BasicBlock> bbiter = bblist.iterator();
        ArrayList<BasicBlock> bbrem = new ArrayList<BasicBlock>();
        while (bbiter.hasNext()) {
            BasicBlock bbl = bbiter.next();
            if (bbl.preds.size() == 1 && bbl.succs.size() == 1) {
                if (bbl.statements.size() == 0) {
                    BasicBlock bpred = bbl.preds.get(0);
                    BasicBlock bsucc = bbl.succs.get(0);
                    bpred.addSuccessor(bsucc);
                    bsucc.addPredecessor(bpred);
                    bpred.succs.remove(bbl);
                    bsucc.preds.remove(bbl);
                    bbl.preds.remove(bpred);
                    bbl.succs.remove(bsucc);
                    bbrem.add(bbl);
                }
            }
        }
        bblist.removeAll(bbrem);
    }

    private void adjustBBforCalls() {
        for (int i = 0; i < bbtoadjust.size(); i++) {
            BasicBlock bbl = bbtoadjust.get(i);
            if (bbl.succs.size() > 1) {
                System.err.println("Too many return points");
                Tools.exit(-1);
            }
            BasicBlock retblock = bbl.succs.get(0);
            bbl.succs.remove(0);
            List l_dir = bbl.statements;
            System.err.println("Adjusting --");
            PrintTools.printlnList(bbl.statements, new PrintWriter(System.err));
            for (int j = 0; j < l_dir.size(); j++) {
                Object o = l_dir.get(j);
                if (!(o instanceof ExpressionStatement)) {
                    System.err.println("This should be a functioncall");
                    break;
                }
                Expression l_ex = ((ExpressionStatement)o).getExpression();
                DepthFirstIterator l_di = new DepthFirstIterator(l_ex);
                while (l_di.hasNext()) {
                    Object l_o = l_di.next();
                    if (!(l_o instanceof FunctionCall)) {
                        continue;
                    }
                    FunctionCall fc = (FunctionCall)l_o;
                    if (fc == null) {
                        break;
                    }
                    ProcBasicBlock pBB = procBB.get(fc.getProcedure());
                    if (pBB != null) {
                        bbl.addSuccessor(pBB.entryBB);
                        pBB.addSuccessor(retblock);
                    }
                }
            }
        }
    }

    private ProcBasicBlock getBBforProc(Procedure proc) {
        //First empty all the stacks
        breakstack.clear();
        continuestack.clear();
        switchstack.clear();
        scopentrystack.clear();
        scopexitstack.clear();
        ProcBasicBlock p_BB = new ProcBasicBlock();
        scopentrystack.push(p_BB.entryBB);
        scopexitstack.push(p_BB.exitBB);
        AnnotationStatement dc = new AnnotationStatement(
                new CommentAnnotation(proc.getName().toString()));
        p_BB.statements.add(dc);
        AnnotationStatement dc1 = new AnnotationStatement(
                new CommentAnnotation("Entry to " + proc.getName().toString()));
        AnnotationStatement dc2 = new AnnotationStatement(
                new CommentAnnotation("RETURN"));
        dc1.setParent(proc);
        dc2.setParent(proc);
        CompoundStatement l_cs = proc.getBody();
        entrybb = p_BB.entryBB;
        exitbb = p_BB.exitBB;
        List<Traversable> arl1 = entrybb.statements;
        arl1.add(dc1);
        arl1 = exitbb.statements;
        arl1.add(dc2);
        currpred = entrybb;
        bblist.add(entrybb);
        bblist.add(exitbb);
        findBBs(l_cs, p_BB.exitBB);
        BasicBlock t_bbl = scopentrystack.pop();
        exitbb.addPredecessor(t_bbl);
        t_bbl.addSuccessor(exitbb);
        return p_BB;
    }

    private void findBBs(Statement stmt, BasicBlock succ) {
        BasicBlock pred = scopentrystack.peek();
        //Expression Statements will be added to this pred block
        //If a new block needs to be created, it will become a successor of pred
        //and be pushed to the top of scopentrystack
        if (stmt instanceof IfStatement) {
            IfStatement ist = (IfStatement)stmt;
            BasicBlock ifentry = new BasicBlock();
            bblist.add(ifentry);
            BasicBlock bblthen = new BasicBlock();
            bblist.add(bblthen);
            BasicBlock ifexit = new BasicBlock();
            bblist.add(ifexit);
            pred.addSuccessor(ifentry);
            ifentry.addPredecessor(pred);
            scopentrystack.push(ifentry);
            scopexitstack.push(ifexit);
            ExpressionStatement l_est =
                    new ExpressionStatement(ist.getControlExpression());
            l_est.setParent(ist);
            if (containsFC(l_est)) {
                BasicBlock condblock = new BasicBlock();
                bblist.add(condblock);
                findBBs(l_est, condblock);
                ifentry = scopentrystack.pop();
                condblock.addPredecessor(ifentry);
                ifentry.addSuccessor(condblock);
                scopentrystack.push(condblock);
            } else {
                ifentry.statements.add(l_est);
            }
            ifentry = scopentrystack.pop();
            ifentry.addSuccessor(bblthen);
            bblthen.addPredecessor(ifentry);
            scopentrystack.push(bblthen);
            findBBs(ist.getThenStatement(), ifexit);
            bblthen = scopentrystack.pop();
            bblthen.addSuccessor(ifexit);
            ifexit.addPredecessor(bblthen);
            if (ist.getElseStatement() != null) {
                BasicBlock bbelse = new BasicBlock();
                bblist.add(bbelse);
                ifentry.addSuccessor(bbelse);
                bbelse.addPredecessor(ifentry);
                scopentrystack.push(bbelse);
                findBBs(ist.getElseStatement(), ifexit);
                bbelse = scopentrystack.pop();
                bbelse.addSuccessor(ifexit);
                ifexit.addPredecessor(bbelse);
            } else {
                //The else path goes directly to ifexit
                ifentry.addSuccessor(ifexit);
                ifexit.addPredecessor(ifentry);
            }
            //A basic block boundary here
            scopentrystack.push(ifexit);
            scopexitstack.pop();
            return;
        }
        if (stmt instanceof ForLoop) {
            ForLoop fl = (ForLoop)stmt;
            BasicBlock loopentry = new BasicBlock();
            bblist.add(loopentry);
            BasicBlock loopexit = new BasicBlock();
            bblist.add(loopexit);
            BasicBlock loopcond = new BasicBlock();
            bblist.add(loopcond);
            BasicBlock loopbody = new BasicBlock();
            bblist.add(loopbody);
            BasicBlock loopstep = new BasicBlock();
            bblist.add(loopstep);
            breakstack.push(loopexit);
            continuestack.push(loopstep);
            loopentry.addPredecessor(pred);
            pred.addSuccessor(loopentry);
            Statement l_init = fl.getInitialStatement();
            Statement l_cond = new ExpressionStatement(fl.getCondition());
            l_cond.setParent(fl);
            Statement l_step = new ExpressionStatement(fl.getStep());
            l_step.setParent(fl);
            Statement l_body = fl.getBody();
            scopentrystack.push(loopentry);
            scopexitstack.push(loopexit);
            findBBs(l_init, loopcond);
            loopentry = scopentrystack.peek();
            loopentry.addSuccessor(loopcond);
            loopcond.addPredecessor(loopentry);
            scopentrystack.push(loopcond);
            findBBs(l_cond, loopbody);
            loopentry = scopentrystack.peek();
            loopentry.addSuccessor(loopbody);
            loopbody.addPredecessor(loopentry);
            scopentrystack.push(loopbody);
            findBBs(l_body, loopexit);
            loopbody = scopentrystack.peek();
            loopbody.addSuccessor(loopstep);
            loopstep.addPredecessor(loopbody);
            scopentrystack.push(loopstep);
            findBBs(l_step, loopexit);
            BasicBlock bbl = scopentrystack.peek();
            bbl.addSuccessor(loopcond);
            loopcond.addPredecessor(bbl);
            loopexit.addPredecessor(loopentry);
            loopentry.addSuccessor(loopexit);
            scopentrystack.push(loopexit);
            breakstack.pop();
            continuestack.pop();
            return;
        }
        if (stmt instanceof DoLoop) {
            DoLoop fl = (DoLoop)stmt;
            BasicBlock loopentry = new BasicBlock();
            bblist.add(loopentry);
            BasicBlock loopexit = new BasicBlock();
            bblist.add(loopexit);
            BasicBlock loopcond = new BasicBlock();
            bblist.add(loopcond);
            BasicBlock bbl;
            continuestack.push(loopcond);
            breakstack.push(loopexit);
            Statement l_cond = new ExpressionStatement(fl.getCondition());
            l_cond.setParent(fl);
            Statement l_body = fl.getBody();
            loopentry.addPredecessor(pred);
            pred.addSuccessor(loopentry);
            scopentrystack.push(loopentry);
            scopexitstack.push(loopexit);
            findBBs(l_body, loopcond);
            bbl = scopentrystack.pop();
            bbl.addSuccessor(loopcond);
            loopcond.addPredecessor(bbl);
            scopentrystack.push(loopcond);
            findBBs(l_cond, loopexit);
            bbl = scopentrystack.pop();
            bbl.addSuccessor(loopentry);
            loopentry.addPredecessor(bbl);
            bbl.addSuccessor(loopexit);
            loopexit.addPredecessor(bbl);
            scopentrystack.push(loopexit);
            continuestack.pop();
            breakstack.pop();
            return;
        }
        if (stmt instanceof WhileLoop) {
            WhileLoop fl = (WhileLoop)stmt;
            BasicBlock loopentry = new BasicBlock();
            bblist.add(loopentry);
            BasicBlock loopexit = new BasicBlock();
            bblist.add(loopexit);
            BasicBlock loopcond = new BasicBlock();
            bblist.add(loopcond);
            BasicBlock bbl;
            DeclarationStatement l_dc;
            continuestack.push(loopcond);
            breakstack.push(loopexit);
            Statement l_cond = new ExpressionStatement(fl.getCondition());
            l_cond.setParent(fl);
            Statement l_body = fl.getBody();
            loopcond.addPredecessor(pred);
            pred.addSuccessor(loopcond);
            scopentrystack.push(loopcond);
            scopexitstack.push(loopexit);
            findBBs(l_cond, loopexit);
            bbl = scopentrystack.pop();
            bbl.addSuccessor(loopentry);
            loopentry.addPredecessor(bbl);
            bbl.addSuccessor(loopexit);
            loopexit.addPredecessor(bbl);
            scopentrystack.push(loopentry);
            findBBs(l_body, loopcond);
            bbl = scopentrystack.pop();
            bbl.addSuccessor(loopcond);
            loopcond.addPredecessor(bbl);
            scopentrystack.push(loopexit);
            continuestack.pop();
            breakstack.pop();
            return;
        }
        if (stmt instanceof BreakStatement) {
            if (!breakstack.isEmpty()) {
                BasicBlock bbl = breakstack.peek();
                pred.addSuccessor(bbl);
                bbl.addPredecessor(pred);
                bbl = new BasicBlock();
                bblist.add(bbl);
                scopentrystack.push(bbl);
            } else {
                System.err.println("Illegal break statement at line " +
                                   stmt.where());
            }
            return;
        }
        if (stmt instanceof ContinueStatement) {
            if (!continuestack.isEmpty()) {
                BasicBlock bbl = continuestack.peek();
                pred.addSuccessor(bbl);
                bbl.addPredecessor(pred);
                bbl = new BasicBlock();
                bblist.add(bbl);
                scopentrystack.push(bbl);
            } else {
                System.err.println("Illegal continue statement at line " +
                                   stmt.where());
            }
            return;
        }
        if (stmt instanceof Case) {
            //I guess we need a new basic block for each case block
            //Assuming that the case expression does not contain fcalls
            Statement l_cex =
                new ExpressionStatement(((Case)stmt).getExpression());
            l_cex.setParent(stmt);
            if (containsFC(l_cex)) {
                System.err.println("Function call in case exp, exiting");
                Tools.exit(-1);
            }
            BasicBlock bbcase = new BasicBlock();
            bblist.add(bbcase);
            bbcase.statements.add(l_cex);
            //Check if pred has a predecessor, otherwise set pred to
            //switchstatement
            pred.addSuccessor(bbcase);
            bbcase.addPredecessor(pred);
            if (pred.preds.size() == 0) {
                BasicBlock bbswitch = switchstack.peek();
                pred.addPredecessor(bbswitch);
                bbswitch.addSuccessor(pred);
            }
            scopentrystack.push(bbcase);
            return;
        }
        if (stmt instanceof Default) {
            BasicBlock bbl = new BasicBlock();
            bblist.add(bbl);
            bbl.addPredecessor(pred);
            pred.addSuccessor(bbl);
            if (pred.preds.size() == 0) {
                BasicBlock bbswitch = switchstack.peek();
                pred.addPredecessor(bbswitch);
                bbswitch.addSuccessor(pred);
            }
            scopentrystack.push(bbl);
            return;
        }
        if (stmt instanceof SwitchStatement) {
            SwitchStatement fl = (SwitchStatement)stmt;
            BasicBlock loopcond = new BasicBlock();
            bblist.add(loopcond);
            BasicBlock loopbody = new BasicBlock();
            bblist.add(loopbody);
            BasicBlock loopexit = new BasicBlock();
            bblist.add(loopexit);
            BasicBlock bbl;
            pred.addSuccessor(loopcond);
            loopcond.addPredecessor(pred);
            scopentrystack.push(loopcond);
            scopexitstack.add(loopexit);
            breakstack.push(loopexit);
            ExpressionStatement l_es =
                new ExpressionStatement(fl.getExpression());
            l_es.setParent(fl);
            findBBs(l_es, loopexit);
            bbl = scopentrystack.pop();
            switchstack.push(bbl);
            bbl.addSuccessor(loopbody);
            loopbody.addPredecessor(bbl);
            scopentrystack.push(loopbody);
            findBBs(fl.getBody(), loopexit);
            bbl = scopentrystack.pop();
            bbl.addSuccessor(loopexit);
            loopexit.addPredecessor(bbl);
            scopentrystack.push(loopexit);
            switchstack.pop();
            breakstack.pop();
            return;
        }
        if (stmt instanceof ReturnStatement) {
            ReturnStatement fl = (ReturnStatement)stmt;
            if (fl.getExpression() != null) {
                ExpressionStatement l_est =
                    new ExpressionStatement(fl.getExpression());
                l_est.setParent(fl);
                findBBs(l_est, succ);
            }
            //Must end the basic block here
            BasicBlock bbl = scopentrystack.pop();
            bbl.addSuccessor(exitbb);
            exitbb.addPredecessor(bbl);
            bbl = new BasicBlock();
            bblist.add(bbl);
            scopentrystack.push(bbl);
            return;
        }
        if (stmt instanceof CompoundStatement) {
            List alist = stmt.getChildren();
            for (int i = 0; i < alist.size(); i++) {
                findBBs((Statement)alist.get(i), succ);
            }
            return;
        }
        if (stmt instanceof ExpressionStatement) {
            handleExpressionStmt(((ExpressionStatement)stmt), pred, succ);
            return;
        }
        if (stmt instanceof DeclarationStatement) {
            return;
        }
        System.err.println("######REACHED HAZY ZONE######");
        System.err.print(stmt.toString());
        System.err.println("\n######################");
    }

    private void handleExpressionStmt(ExpressionStatement stmt,
                                      BasicBlock pred, BasicBlock succ) {
        if (containsFC(stmt)) {
            Expression ex = stmt.getExpression();
            List<Traversable> l_li = ex.getChildren();
            BasicBlock retblock = new BasicBlock();
            bblist.add(retblock);
            BasicBlock callblock = new BasicBlock();
            bblist.add(callblock);
            pred.addSuccessor(callblock);
            callblock.addPredecessor(pred);
            callblock.statements.add(stmt);
            bbtoadjust.add(callblock);
            for (int i = 0; i < l_li.size(); i++) {
                Object o = l_li.get(i);
                if (o instanceof FunctionCall) {
                    FunctionCall fc = (FunctionCall)o;
                    if (fc != null) {
                        //Add the functioncall to the proccalls hasmap
                        Procedure l_pr = fc.getProcedure();
                        if (l_pr != null) {
                            if (proccalls.containsKey(l_pr)) {
                                List<BasicBlock> l_ar = proccalls.get(l_pr);
                                l_ar.add(callblock);
                            } else {
                                List<BasicBlock> l_ar =
                                        new ArrayList<BasicBlock>();
                                l_ar.add(callblock);
                                proccalls.put(l_pr, l_ar);
                            }
                        }
                    }
                }
            }
            callblock.addSuccessor(retblock);
            retblock.addPredecessor(callblock);
            scopentrystack.push(retblock);
        } else {
            pred.statements.add(stmt);
        }
    }

    public void print(OutputStream stream) {
        PrintStream p = new PrintStream(stream);
        //Create clusters for all the bblocks in terms of procedures
        HashMap<Procedure, List<Integer>> p_clusters =
                new HashMap<Procedure, List<Integer>>();
        Iterator<Procedure> l_it = procBB.keySet().iterator();
        while (l_it.hasNext()) {
            p_clusters.put(l_it.next(), new ArrayList<Integer>());
        }
        for (int i = 0; i < bblist.size(); i++) {
            Procedure l_pr = getProc(bblist.get(i));
            if (l_pr != null) {
                List<Integer> l_ar = p_clusters.get(l_pr);
                if (l_ar == null) {
                    System.err.println("Null array for proc " + l_pr);
                } else {
                    l_ar.add(new Integer(i));
                }
            }
        }
        p.println("digraph G {");
        Iterator<Procedure> l_piter = p_clusters.keySet().iterator();
        while (l_piter.hasNext()) {
            Procedure l_p = l_piter.next();
            p.println("subgraph cluster" + l_p.getName().toString() + " { ");
            List<Integer> l_bbls = p_clusters.get(l_p);
            for (int j = 0; j < l_bbls.size(); j++) {
                p.print(" node" + l_bbls.get(j) + "; ");
            }
            p.println("label = \"" + l_p.getName().toString() + "\";}\n");
        }
        ByteArrayOutputStream l_bar = new ByteArrayOutputStream();
        for (int i = 0; i < bblist.size(); i++) {
            p.print("node" + i + " [label=\"");
            BasicBlock bbl = bblist.get(i);
            PrintTools.printlnList(bbl.statements, new PrintWriter(l_bar));
            String l_s = l_bar.toString();
            l_s = l_s.replace('{', ' ');
            l_s = l_s.replace('}', ' ');
            l_s = l_s.replaceAll("\n", " \\\\n");
            l_s = l_s.replace('"', '`');
            l_bar.reset();
            p.print(l_s);
            p.println("\"];");
        }
        for (int i = 0; i < bblist.size(); i++) {
            BasicBlock bbl = bblist.get(i);
            for (int j = 0; j < bbl.succs.size(); j++) {
                BasicBlock succ = bbl.succs.get(j);
                int k = bblist.indexOf(succ);
                if (k < 0) {
                    System.err.println("Error, a bblock not found in the list");
                } else {
                    p.print("node" + i + "->" + "node" + k + ";");
                }
            }
        }
        int start = bblist.indexOf(getRoot());
        if (start < 0) {
            start = 0;
        }
        p.println("\nStart->node" + start);
        p.println("Start [shape=Mdiamond] ;");
        p.println("\n}");
    }

    private Procedure getProc(BasicBlock bbl) {
        //If bblock has statements, return their procedure, else return
        // procedure of successor
        if (bbl.statements.size() > 0) {
            Statement l_s = (Statement)bbl.statements.get(0);
            return l_s.getProcedure();
        } else {
            if (bbl.succs.size() < 1) {
                if (bbl.preds.size() > 1) {
                    System.err.println("Procedure of a bblock not found");
                    return null;
                } else {
                    return getProc(bbl.preds.get(0));
                }
            } else {
                return getProc(bbl.succs.get(0));
            }
        }
    }
}
