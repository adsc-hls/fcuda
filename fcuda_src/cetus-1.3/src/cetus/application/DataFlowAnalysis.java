package cetus.application;

import cetus.analysis.AnalysisPass;
import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.SwitchStatement;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class supports the basic data flow analysis. It supports backward
 * and forward analysis. It also supports may and must analysis.
 * Any subclass which implements a specific analysis should implement
 * the abstract methods for creating the analysis target set and
 * for extracting the GEN set and KILL set.
 * DataFlowAnalysis implements the iterative worklist algorithm for the analysis.
 *
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public abstract class DataFlowAnalysis extends AnalysisPass implements Runnable {

    // Control Flow Graph
    protected Map<Procedure, CFGraph> cfgMap;
    // Definition Mapping Table
    protected Map<Procedure, Set<AnalysisTarget>> analysisTargetMap;
    // Is this forward analysis ?
    private boolean forward;
    // Is this may analysis ?
    private boolean may;
    // current analysis target
    protected Procedure targetProc;
    // 
    enum GenOption {GENONLY, GENKILL};

    public DataFlowAnalysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            boolean forward,
            boolean may) {
        super(program);
        this.cfgMap = cfgMap;
        this.analysisTargetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        this.forward = forward;
        this.may = may;
    }

    @Override
    public String getPassName() {
        return "[DATA-FLOW-ANALYSIS]";
    }

    @Override
    public void start() {
        Set<Procedure> proc_list = ChainTools.getProcedureSet(program);
        for (Procedure proc : proc_list) {
            compute(proc);
        }
    }
    
    abstract Set<AnalysisTarget> createAnalysisTargets(Procedure proc);

    protected void initializeGenKillBitSet(Procedure proc, int bitNum) {
        Iterator cfgIter = cfgMap.get(proc).iterator();
        while (cfgIter.hasNext()) {
            DFANode cfgNode = (DFANode) cfgIter.next();
            BitSet genBitSet = new BitSet(bitNum);
            cfgNode.putData("GenSet", genBitSet);
            BitSet killBitSet = new BitSet(bitNum);
            cfgNode.putData("KillSet", killBitSet);
        }
    }

    abstract void extractGenKillSet(Set<AnalysisTarget> targetSet, Procedure proc);

    private void printDataFlowInformation(CFGraph cfg) {
        Iterator cfgIter = cfg.iterator();
        while (cfgIter.hasNext()) {
            DFANode node = (DFANode) cfgIter.next();
            if (CFGraph.getIR(node) instanceof SwitchStatement) {
                System.out.println("Code : " + node.getData("ir").toString().substring(0, 9));
            } else {
                System.out.println("Code : " + node.getData("ir"));
            }
            System.out.println("  IN   BitSet : " + node.getData("InSet").toString());
            System.out.println("  GEN  BitSet : " + node.getData("GenSet").toString());
            System.out.println("  KILL BitSet : " + node.getData("KillSet").toString());
            System.out.println("  OUT  BitSet : " + node.getData("OutSet").toString());
            System.out.println("  Pred.size: " + node.getPreds().size() + ", Succ.size: " + node.getSuccs().size());
        }
    }

    protected void calculateDataFlowEquation(Procedure proc, BitSet initSet) {
        if (forward) {
            calculateForwardDataFlowEquation(proc, initSet);
        } else {
            calculateBackwardDataFlowEquation(proc, initSet);
        }
    }

    protected void calculateCombine(BitSet result, BitSet input) {
        if (may) {
            result.or(input);
        } else {
            result.and(input);
        }
    }

    protected void calculateFunction(BitSet result, BitSet input, BitSet genSet, BitSet killSet) {
        result.or(input);
        result.andNot(killSet);
        result.or(genSet);
    }

    protected void calculateForwardDataFlowEquation(Procedure proc, BitSet initSet) {
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("## Forward Analysis.... for " + proc.getSymbolName());
        // OUT = GEN U (IN - KILL)
        LinkedList<DFANode> workList = new LinkedList<DFANode>();
        Iterator cfgIter = cfgMap.get(proc).iterator();
        DFANode entryNode = cfgMap.get(proc).getEntry();
        // add all the node to worklist
        while (cfgIter.hasNext()) {
            DFANode dfaNode = (DFANode) cfgIter.next();
            dfaNode.putData("OutSet", new BitSet(initSet.size()));
            workList.add(dfaNode);
        }
        // set entry node with init data
        entryNode.putData("InSet", (BitSet) initSet.clone());
        workList.remove(entryNode);

        while (workList.isEmpty() == false) {
            DFANode currentNode = workList.removeFirst();
            BitSet previousOutSet = (BitSet) ((BitSet) currentNode.getData("OutSet")).clone();
            BitSet inSet = new BitSet(initSet.size());
            if (may == false) { // must
                inSet.flip(0, initSet.size());
            }
            for (DFANode pred : currentNode.getPreds()) {
                calculateCombine(inSet, (BitSet) pred.getData("OutSet"));
            }
            BitSet outSet = (BitSet) currentNode.getData("OutSet");
            outSet.clear();
            //
            calculateFunction(outSet, inSet, (BitSet) currentNode.getData("GenSet"), (BitSet) currentNode.getData("KillSet"));
            //
            currentNode.putData("InSet", inSet);
            currentNode.putData("OutSet", outSet);

            if (previousOutSet.equals(outSet) == false) {
                workList.addAll(currentNode.getSuccs());
            }
        }
        cleanupUnnecessaryData();
//        printDataFlowInformation(cfgMap.get(proc));
    }
    
    abstract void cleanupUnnecessaryData();

    protected void calculateBackwardDataFlowEquation(Procedure proc, BitSet initSet) {
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("## Backward Analysis... for " + proc.getSymbolName());
        // IN = GEN U (OUT - KILL)
        LinkedList<DFANode> workList = new LinkedList<DFANode>();
        Iterator cfgIter = cfgMap.get(proc).iterator();
        // add all node to the worklist
        while (cfgIter.hasNext()) {
            DFANode dfaNode = (DFANode) cfgIter.next();
            dfaNode.putData("InSet", new BitSet(initSet.size()));
            workList.addFirst(dfaNode);
        }
        // set exit node with init data
        List<DFANode> exitNodeList = cfgMap.get(proc).getExitNodes();
        for (DFANode exitNode : exitNodeList) {
            exitNode.putData("OutSet", (BitSet) initSet.clone());
            workList.remove(exitNode);
        }

        while (workList.isEmpty() == false) {
            DFANode currentNode = workList.removeFirst();
            BitSet previousInSet = (BitSet) ((BitSet) currentNode.getData("InSet")).clone();
            BitSet outSet = (BitSet) initSet.clone();
            if (may == false) { // must
                outSet.flip(0, initSet.size());
            }
            for (DFANode succ : currentNode.getSuccs()) {
                calculateCombine(outSet, (BitSet) succ.getData("InSet"));
            }
            BitSet inSet = (BitSet) currentNode.getData("InSet");
            inSet.clear();
            //
            calculateFunction(inSet, outSet, (BitSet) currentNode.getData("GenSet"), (BitSet) currentNode.getData("KillSet"));
            currentNode.putData("OutSet", outSet);
            currentNode.putData("InSet", inSet);
            if (previousInSet.equals(inSet) == false) {
                workList.addAll(currentNode.getPreds());
            }
        }
        // delete unnecessary data
        cfgIter = cfgMap.get(proc).iterator();
        while (cfgIter.hasNext()) {
            DFANode dfaNode = (DFANode) cfgIter.next();
            dfaNode.removeData("InSet");
            dfaNode.removeData("KillSet");
            dfaNode.removeData("GenSet");
        }
//        printDataFlowInformation(cfgMap.get(proc));
    }

    protected void compute(Procedure proc) {
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("############## Executing Data Flow Analysis " + getPassName() + " for proc: " + proc.getSymbolName());
        Set<AnalysisTarget> targetList = createAnalysisTargets(proc);
        analysisTargetMap.put(proc, targetList);
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("AnalysisTarget size: " + targetList.size());
        initializeGenKillBitSet(proc, targetList.size());
        extractGenKillSet(targetList, proc);
        BitSet initBitSet = new BitSet(targetList.size());
        calculateDataFlowEquation(proc, initBitSet);
        // Thread Version
        if (targetMap != null) {
            synchronized (targetMap) {
            	targetMap.put(proc, targetList);
            }
        }
    }

    public Map<Procedure, Set<AnalysisTarget>> getAnalysisTargetListMap() {
        return analysisTargetMap;
    }

    public void run() {
    	compute(targetProc);
    }
    
    Map<Procedure, Set<AnalysisTarget>> targetMap;
    public void setAnalysisTarget(Procedure targetProc, Map<Procedure, Set<AnalysisTarget>> targetMap) {
    	this.targetProc = targetProc;
    	this.targetMap = targetMap;
    }
}
