package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.hir.FunctionCall;
import cetus.hir.IRTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Traversable;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class performs the regional kill set analysis for program summary graph.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class PSGRegionalKillSetAnlaysis extends DataFlowAnalysis {

    private Set<Procedure> globalProcSet;       // proc set directly or indirectly use/def global variables

    public PSGRegionalKillSetAnlaysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            Map<Procedure, Set<AnalysisTarget>> targetSetMap,
            Set<Procedure> globalProcSet) {
        super(program, cfgMap, true, false); // cfgMap should be from ReachingDefinitionAnalysis for global variable
        this.analysisTargetMap = targetSetMap;
        this.globalProcSet = globalProcSet;
    }

    @Override
    public String getPassName() {
        return "[PSG-REGIONAL-KILLSET-ANALYSIS]";
    }

    @Override
    protected void initializeGenKillBitSet(Procedure proc, int bitNum) {
        // do nothing, keep the previous bit set
    }

    Set<AnalysisTarget> createAnalysisTargets(Procedure proc) {
        // do nothing
        return analysisTargetMap.get(proc);
    }

    @Override
    void extractGenKillSet(Set<AnalysisTarget> listDefMapping, Procedure proc) {
        // do nothing, already set in the control flow graph
        CFGraph cfg = cfgMap.get(proc);
        Iterator cfgIter = cfg.iterator();
        while (cfgIter.hasNext()) {
            DFANode node = (DFANode) cfgIter.next();
            Traversable currentIR = (Traversable) CFGraph.getIR(node);
            if (currentIR == null) {
                continue;
            }
            List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
            if (fcList == null) {
                continue;
            }
            for (FunctionCall fc : fcList) {
                Procedure callee = fc.getProcedure();
                if (callee == null) {
                    continue;
                }
                if (globalProcSet.contains(callee) == false) {
                    continue;
                }
                BitSet genBitSet = node.getData("GenSet");
                for (int i = 0; i < listDefMapping.size(); i++) {
                    genBitSet.set(i);
                }
                node.putData("GenSet", genBitSet);
            }
        }
        return;
    }

    @Override
    protected void calculateFunction(BitSet result, BitSet input, BitSet genSet, BitSet killSet) {
        result.or(input);
        result.or(killSet);
        result.andNot(genSet);
    }

	@Override
	void cleanupUnnecessaryData() {
        // delete unnecessary data
        Iterator cfgIter = cfgMap.get(targetProc).iterator();
        while (cfgIter.hasNext()) {
            DFANode dfaNode = (DFANode) cfgIter.next();
            dfaNode.removeData("OutSet");
            dfaNode.removeData("KillSet");
            dfaNode.removeData("GenSet");
        }
	}

}
