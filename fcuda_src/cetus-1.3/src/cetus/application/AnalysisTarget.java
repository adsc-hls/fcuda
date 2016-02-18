package cetus.application;

import cetus.analysis.DFANode;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IRTools;
import cetus.hir.Procedure;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * This class contains the information for the target of the data flow analysis.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class AnalysisTarget {
    // Container for Satellite Data

    private Map<String, Object> data;

    public AnalysisTarget(Expression expression, DFANode cfgNode, Procedure proc) {
        data = new HashMap<String, Object>();
        if (ChainTools.isStructureAccess(expression, proc)) {
            data.put("expression", ChainTools.getIDVariablePlusMemberInStruct(expression));
        } else {
            if (ChainTools.isArrayAccessWithConstantIndex(expression)) {
                data.put("expression", expression);
            } else {
//                Expression idEx = ChainTools.getVarIDInMixedForm(expression, proc);
//                if (idEx == null) {
//                    throw new RuntimeException("Expression is null for AnalysisTarget: " + expression + ", proc: " + IRTools.getParentProcedure(idEx));
//                }
//                data.put("expression", ChainTools.getVarIDInMixedForm(expression, proc));
                Expression idEx = ChainTools.getRefIDExpression(expression, proc);
                if (idEx == null) {
                    throw new RuntimeException("Expression is null for AnalysisTarget: " + expression + ", proc: " + IRTools.getParentProcedure(idEx));
                }
                data.put("expression", idEx);
            }
        }
        data.put("node", cfgNode);
        data.put("proc", proc);
        if (cfgNode == null) {
            throw new RuntimeException("DFANode is null for " + expression);
        }
    }

    public AnalysisTarget(FunctionCall fc, DFANode cfgNode, Procedure proc) {
        data = new HashMap<String, Object>();
        data.put("function_call", fc);
        data.put("node", cfgNode);
        data.put("proc", proc);
        if (cfgNode == null) {
            throw new RuntimeException("DFANode is null for " + fc);
        }
    }

    public Expression getExpression() {
        return (Expression) data.get("expression");
    }

    public Procedure getProcedure() {
        return (Procedure) data.get("proc");
    }

    public void setDummy() {
        data.put("dummy", true);
    }

    public DFANode getDFANode() {
        return (DFANode) data.get("node");
    }

    public FunctionCall getFunctionCall() {
        return (FunctionCall) data.get("function_call");
    }

    public void putUseChain(Set<AnalysisTarget> useChain) {
        data.put("use_chain", useChain);
    }

    public void putDefChain(Set<AnalysisTarget> defChain) {
        data.put("def_chain", defChain);
    }

    public Set<AnalysisTarget> getUseChain() {
        return (Set<AnalysisTarget>) data.get("use_chain");
    }

    public Set<AnalysisTarget> getDefChain() {
        return (Set<AnalysisTarget>) data.get("def_chain");
    }

    public void addUseChain(AnalysisTarget use) {
        if (this.getDFANode().getData("param") != null) {
            if (this.getProcedure().equals(use.getProcedure()) == false) {
                return;
            }
        }
        Set<AnalysisTarget> chain = getUseChain();
        if (chain == null) {
            chain = new HashSet<AnalysisTarget>();
            putUseChain(chain);
        }
        chain.add(use);
    }

    public void addDefChain(AnalysisTarget def) {
        Set<AnalysisTarget> chain = getDefChain();
        if (chain == null) {
            chain = new HashSet<AnalysisTarget>();
            putDefChain(chain);
        }
        chain.add(def);
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || this.getClass() != o.getClass()) {
            return false;
        }
        if (getExpression() != null) {
            if (((AnalysisTarget) o).getExpression() == null) {
                return this.getExpression() == null;
            }
            return System.identityHashCode(((AnalysisTarget) o).getExpression()) ==
                    System.identityHashCode(this.getExpression());
        } else if (getFunctionCall() != null) {
            if (((AnalysisTarget) o).getFunctionCall() == null) {
                return this.getFunctionCall() == null;
            }
            return System.identityHashCode(((AnalysisTarget) o).getFunctionCall()) ==
                    System.identityHashCode(this.getFunctionCall());
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        int hash = 3;
        return hash;
    }
}
