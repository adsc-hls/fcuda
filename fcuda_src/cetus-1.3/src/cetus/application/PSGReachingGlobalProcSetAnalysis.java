package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.hir.FunctionCall;
import cetus.hir.IRTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Traversable;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class generates connection information between function calls containing global variables.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class PSGReachingGlobalProcSetAnalysis extends DataFlowAnalysis {

    private Set<Procedure> globalProcSet;       // proc set directly or indirectly use/def global variables

    public PSGReachingGlobalProcSetAnalysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            Set<Procedure> globalProcSet) {
        super(program, cfgMap, true, true); // cfgMap should be from ReachingDefinitionAnalysis for global variable
        this.globalProcSet = globalProcSet;
    }

    @Override
    public String getPassName() {
        return "[PSG-REACHING-GLOBAL-PROC-SET-ANALYSIS]";
    }

    Set<AnalysisTarget> createAnalysisTargets(Procedure proc) {
        // create AnalysisTarget (FunctionCall of GlobalProcSet)
        Set<AnalysisTarget> targetSet = new HashSet<AnalysisTarget>();
        CFGraph cfg = cfgMap.get(proc);
        Iterator cfgIter = cfg.iterator();
        while (cfgIter.hasNext()) {
            DFANode node = (DFANode) cfgIter.next();
            Traversable currentIR = (Traversable) CFGraph.getIR(node);
            List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
            if (fcList == null || fcList.size() == 0) {
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
                targetSet.add(new AnalysisTarget(fc, node, proc));
            }
        }
        return targetSet;
    }

    @Override
    void extractGenKillSet(Set<AnalysisTarget> listDefMapping, Procedure proc) {
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
                BitSet killBitSet = node.getData("KillSet");
                AnalysisTarget[] targetArray = new AnalysisTarget[listDefMapping.size()];
                listDefMapping.toArray(targetArray);
                for (int idx = 0; idx < targetArray.length; idx++) {
                    AnalysisTarget target = targetArray[idx];
                    FunctionCall targetFC = target.getFunctionCall();
                    if (System.identityHashCode(targetFC) ==
                            System.identityHashCode(fc)) {
                        genBitSet.set(idx);
                    }
                    if (globalProcSet.contains(targetFC.getProcedure())) {
                        killBitSet.set(idx);
                    }
                }
                node.putData("GenSet", genBitSet);
                node.putData("KillSet", killBitSet);
            }
        }
        return;
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
