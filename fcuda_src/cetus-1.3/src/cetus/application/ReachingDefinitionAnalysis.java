package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.hir.ConditionalExpression;
import cetus.hir.DataFlowTools;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.NestedDeclarator;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.StandardLibrary;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclarator;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class performs the reaching definition analysis.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class ReachingDefinitionAnalysis extends DataFlowAnalysis {

    private LinkedHashSet<AnalysisTarget> globalDefList;

    public ReachingDefinitionAnalysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            LinkedHashSet<AnalysisTarget> globalDefList) {
        super(program, cfgMap, true, true);
        this.analysisTargetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        this.globalDefList = globalDefList;
    }

    @Override
    public String getPassName() {
        return "[REACHING-DEFINITION-ANALYSIS]";
    }

    Set<AnalysisTarget> createAnalysisTargets(Procedure proc) {
        Set<AnalysisTarget> listDefMapping = new LinkedHashSet<AnalysisTarget>();
        Iterator cfgIter = cfgMap.get(proc).iterator();
        while (cfgIter.hasNext()) {
            DFANode cfgNode = (DFANode) cfgIter.next();
            Object currentIR = CFGraph.getIR(cfgNode);
            if (currentIR == null) {
                // The first cfgNode in CFG is always null. Skip this cfgNode.
                continue;
            }
            if (currentIR instanceof UnaryExpression) {
                UnaryExpression unaryEx = (UnaryExpression) currentIR;
                UnaryOperator uop = unaryEx.getOperator();
                if (uop == UnaryOperator.POST_DECREMENT ||
                        uop == UnaryOperator.POST_INCREMENT ||
                        uop == UnaryOperator.PRE_DECREMENT ||
                        uop == UnaryOperator.PRE_INCREMENT) {
                    listDefMapping.add(new AnalysisTarget(unaryEx, cfgNode, proc));
                }
            } else if (currentIR instanceof VariableDeclarator) {
                if (cfgNode.getData("param") != null) {
                    listDefMapping.add(new AnalysisTarget(((VariableDeclarator) cfgNode.getData("ir")).getID(), cfgNode, proc));
                } else {
                    List<Expression> defSet = ChainTools.getDefListInDec((VariableDeclarator) currentIR);
                    for (Expression definedExpression : defSet) {
                        listDefMapping.add(new AnalysisTarget(definedExpression, cfgNode, proc));
                    }
                }
            } else if (currentIR instanceof NestedDeclarator) {
                if (cfgNode.getData("param") != null) {
                    listDefMapping.add(new AnalysisTarget(((NestedDeclarator) cfgNode.getData("ir")).getID(), cfgNode, proc));
                } else {
                    List<Expression> defSet = ChainTools.getDefListInNestedDec((NestedDeclarator) currentIR);
                    for (Expression definedExpression : defSet) {
                        listDefMapping.add(new AnalysisTarget(definedExpression, cfgNode, proc));
                    }
                }
            } else if (currentIR instanceof SwitchStatement) {
                // Nothint to do on SwitchStatement
            } else if (currentIR instanceof Traversable) {
                Traversable traversableStmt = (Traversable) currentIR;
                addPlainDefSetToDefMappingTable(listDefMapping, traversableStmt, cfgNode, proc);
                if (IRTools.containsFunctionCall(traversableStmt)) {
                    addSideEffectStdLibParameterToDefMappingTable(listDefMapping, traversableStmt, cfgNode, proc);
                }
            } else {
                throw new RuntimeException("Unexpected Statement: IR: " + currentIR.toString() + ", Proc: " + proc.getSymbolName() + ", Class: " + currentIR.getClass().getCanonicalName());
            }
        }
//        printDefinitionMappingTable(listDefMapping, proc);
        return listDefMapping;
    }

    void extractGenKillSet(Set<AnalysisTarget> listDefMapping, Procedure proc) {
        AnalysisTarget defMapEntry[] = new AnalysisTarget[listDefMapping.size()];
        listDefMapping.toArray(defMapEntry);
        Iterator cfgIter = cfgMap.get(proc).iterator();
        while (cfgIter.hasNext()) {
            DFANode cfgNode = (DFANode) cfgIter.next();
            Object currentIR = CFGraph.getIR(cfgNode);
            if (currentIR == null) {
                continue;
            }
            if (currentIR instanceof UnaryExpression) {
                UnaryExpression unaryEx = (UnaryExpression) currentIR;
                UnaryOperator unaryOp = unaryEx.getOperator();
                if (unaryOp == UnaryOperator.POST_DECREMENT ||
                        unaryOp == UnaryOperator.POST_INCREMENT ||
                        unaryOp == UnaryOperator.PRE_DECREMENT ||
                        unaryOp == UnaryOperator.PRE_INCREMENT) {
                    Expression myEx = unaryEx.getExpression();
                    performGenKillSetting(cfgNode, defMapEntry, myEx, currentIR, proc, GenOption.GENKILL);
                } else {
                    Traversable traversableStmt = (Traversable) currentIR;
//                    Set<Expression> defSet = DataFlowTools.getDefSet(traversableStmt);
                    List<Expression> defList = ChainTools.getPlainDefList(traversableStmt, proc);
                    // Set GEN and KILL for Def Set for the procedure
                    // handle the case of assignment "var = xxx"
                    for (Expression defEx : defList) {
                        performGenKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc, GenOption.GENKILL);
                    }
                    extractGenKillSetForLibraryCalls(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
                    extractKillSetFromSideEffectFunctions(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
                }
            } else if (currentIR instanceof VariableDeclarator) {
                if (cfgNode.getData("param") != null) {
//                    System.out.println("param gen: " + ((VariableDeclarator) cfgNode.getData("ir")).getID());
                    performGenKillSetting(cfgNode, defMapEntry, ((VariableDeclarator) cfgNode.getData("ir")).getID(), currentIR, proc, GenOption.GENKILL);
                } else {
                    List<Expression> defSet = ChainTools.getDefListInDec((VariableDeclarator) currentIR);
                    for (Expression defEx : defSet) {
                        performGenKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc, GenOption.GENKILL);
                    }
                }
                extractGenKillSetForLibraryCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
                extractKillSetFromSideEffectFunctions((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
            } else if (currentIR instanceof NestedDeclarator) {
                if (cfgNode.getData("param") != null) {
//                    System.out.println("param gen: " + ((VariableDeclarator) cfgNode.getData("ir")).getID());
                    performGenKillSetting(cfgNode, defMapEntry, ((NestedDeclarator) cfgNode.getData("ir")).getID(), currentIR, proc, GenOption.GENKILL);
                } else {
                    List<Expression> defSet = ChainTools.getDefListInNestedDec((NestedDeclarator) currentIR);
                    for (Expression defEx : defSet) {
                        performGenKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc, GenOption.GENKILL);
                    }
                }
                extractGenKillSetForLibraryCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
                extractKillSetFromSideEffectFunctions((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
            } else if (currentIR instanceof SwitchStatement) {
                // nothing to do on SwitchStatement
            } else if (currentIR instanceof Traversable) {
                Traversable traversableStmt = (Traversable) currentIR;
                List<Expression> defSet = ChainTools.getPlainDefList(traversableStmt, proc);//DataFlowTools.getDefList(traversableStmt);
                // Set GEN and KILL for Def Set for the procedure
                // handle the case of assignment "var = xxx"
                for (Expression defEx : defSet) {
                    performGenKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc, GenOption.GENKILL);
                }
                extractGenKillSetForLibraryCalls(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
                extractKillSetFromSideEffectFunctions(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
            } else {
                throw new RuntimeException("Unexpected Statement IR: " + currentIR.toString() + ", Proc: " + proc.getSymbolName() + ", Class: " + currentIR.getClass().getCanonicalName());
            }
        }
    }

    private void addPlainDefSetToDefMappingTable(Set<AnalysisTarget> listDefMapping, Traversable tr, DFANode node, Procedure proc) {
        List<Expression> defList = ChainTools.getPlainDefList(tr, proc);//DataFlowTools.getDefList(tr);
        // Add var to the def mapping table if it has the form of "var = ..."
        for (Expression myEx : defList) {
            listDefMapping.add(new AnalysisTarget(myEx, node, proc));
        }
    }

    private void addSideEffectStdLibParameterToDefMappingTable(Set<AnalysisTarget> listDefMapping, Traversable traversableStmt, DFANode cfgNode, Procedure proc) {
        List<FunctionCall> funcCalls = IRTools.getFunctionCalls(traversableStmt);
        for (FunctionCall funcC : funcCalls) {
            int[] paramIdxSet = null;
            if (ThirdPartyLibrary.hasSideEffectOnParameter(funcC)) {
                paramIdxSet = ThirdPartyLibrary.getSideEffectParamIndices(funcC);
                if (paramIdxSet == null) {
                    throw new RuntimeException("Side Effect Para Indices should be set for " + funcC.getName());
                }
            } else if (StandardLibrary.hasSideEffectOnParameter(funcC)) {
                paramIdxSet = StandardLibrary.getSideEffectParamIndices(funcC);
                if (paramIdxSet == null) {
                    throw new RuntimeException("Side Effect Para Indices should be set for " + funcC.getName());
                }
            }
            if (paramIdxSet != null) {
                for (int paramIdx : paramIdxSet) {
                    if (paramIdx < funcC.getNumArguments()) {
                        Expression sideEffectParam = funcC.getArgument(paramIdx);
                        List<Expression> defAddList = new ArrayList<Expression>();
                        if (sideEffectParam instanceof ConditionalExpression) {
                        	defAddList.addAll(ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)sideEffectParam, proc));
                        } else {
                            defAddList.add(sideEffectParam);
                        }
                        for (Expression defToTest : defAddList) {
                        	Expression refEx = ChainTools.getRefIDExpression(defToTest, proc);
                        	if (refEx != null) {
                                listDefMapping.add(new AnalysisTarget(refEx, cfgNode, proc));
                        	}
//                            sideEffectParam = defToTest;
//                            if (sideEffectParam instanceof UnaryExpression) {
//                                sideEffectParam = ((UnaryExpression) sideEffectParam).getExpression();
//                            }
//                            Expression pEx = ChainTools.getVarIDInMixedForm(sideEffectParam, proc);
//                            if (pEx != null) {
//                                sideEffectParam = pEx;
//                            }
//                            if (ChainTools.countIDEx(sideEffectParam) > 1) {
//                                List<Expression> defList = ChainTools.getUseList(sideEffectParam);
//                                for (Expression def : defList) {
//                                    if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                        listDefMapping.add(new AnalysisTarget(def, cfgNode, proc));
//                                    }
//                                }
//                            } else {
//                                if (ChainTools.getIDExpression(sideEffectParam) != null) {
//                                    listDefMapping.add(new AnalysisTarget(sideEffectParam, cfgNode, proc));
//                                }
//                            }
                        }
                    }
                }
            }
//            if (paramIdxSet != null) {
//                for (int paramIdx : paramIdxSet) {
//                    if (paramIdx < funcC.getNumArguments()) {
//                        Expression sideEffectParam = funcC.getArgument(paramIdx);
//                        List<Expression> defAddList = new ArrayList<Expression>();
//                        if (sideEffectParam instanceof ConditionalExpression) {
//                            ConditionalExpression condEx = (ConditionalExpression) sideEffectParam;
//                            defAddList.add(condEx.getTrueExpression());
//                            defAddList.add(condEx.getFalseExpression());
//                        } else {
//                            defAddList.add(sideEffectParam);
//                        }
//                        for (Expression defToTest : defAddList) {
//                            sideEffectParam = defToTest;
//                            if (sideEffectParam instanceof UnaryExpression) {
//                                sideEffectParam = ((UnaryExpression) sideEffectParam).getExpression();
//                            }
////                            Expression idEx = ChainTools.getIDExpression(sideEffectParam);
//                            Expression pEx = ChainTools.getVarIDInMixedForm(sideEffectParam, proc);
//                            if (pEx != null) {
//                                sideEffectParam = pEx;
//                            }
//                            if (ChainTools.countIDEx(sideEffectParam) > 1) {
//                                List<Expression> defList = ChainTools.getUseList(sideEffectParam);
//                                for (Expression def : defList) {
//                                    if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                        listDefMapping.add(new AnalysisTarget(def, cfgNode, proc));
//                                    }
//                                }
//                            } else {
//                                if (ChainTools.getIDExpression(sideEffectParam) != null) {
//                                    listDefMapping.add(new AnalysisTarget(sideEffectParam, cfgNode, proc));
//                                }
//                            }
//                        }
//                    }
//                }
//            }
        }
    }

    private void printDefinitionMappingTable(Set<AnalysisTarget> defList, Procedure proc) {
        System.out.println("Def Mapping Table for : " + proc.getSymbolName());
        int idx = 0;
        for (AnalysisTarget def : defList) {
            System.out.println("DEF[" + idx++ + "] = " + def.getExpression() + ", DFANode: " + CFGraph.getIR(def.getDFANode()));
        }
    }

    private void performGenKillSetting(
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            Expression expression,
            Object currentIR,
            Procedure proc,
            GenOption genOption) {
        Expression expToProcess = null;
        if (ChainTools.containsIDEx(expression) == false) {
//            System.out.println("No ID expression in : " + expression);
            return;
        }
        if (ChainTools.isArrayAccessWithConstantIndex(expression)) {
            expToProcess = expression;
        } else {
            if (ChainTools.isStructureAccess(expression, proc)) {
                expToProcess = ChainTools.getIDVariablePlusMemberInStruct(expression);
            } else {
                expToProcess = ChainTools.getRefIDExpression(expression, proc); //getIDExpression(expression);
            }
        }
        ChainTools.setGenBit(cfgNode, defMapEntry, expToProcess);
        ChainTools.setKillBit(cfgNode, defMapEntry, expToProcess, proc);
        if (currentIR instanceof Statement) {
            ChainTools.setKillBitForAlias(cfgNode, defMapEntry, expToProcess, (Statement) currentIR, proc);
        }
    }

    private void extractGenKillSetForLibraryCalls(
            Traversable traversableStmt,
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            Object currentIR,
            Procedure proc) {
        if (IRTools.containsFunctionCall(traversableStmt)) {
            List<FunctionCall> funcCalls = IRTools.getFunctionCalls(traversableStmt);
            for (FunctionCall funcC : funcCalls) {
                int[] paramIdxSet = null;
                if (ThirdPartyLibrary.hasSideEffectOnParameter(funcC)) {
                    paramIdxSet = ThirdPartyLibrary.getSideEffectParamIndices(funcC);
                    if (paramIdxSet == null) {
                        throw new RuntimeException("Side Effect Para Indices should be set for " + funcC.getName());
                    }
                } else if (StandardLibrary.hasSideEffectOnParameter(funcC)) {
                    paramIdxSet = StandardLibrary.getSideEffectParamIndices(funcC);
                    if (paramIdxSet == null) {
                        throw new RuntimeException("Side Effect Para Indices should be set for " + funcC.getName());
                    }
                }
                if (paramIdxSet != null) {
                    for (int paramIdx : paramIdxSet) {
                        if (paramIdx < funcC.getNumArguments()) {
                            Expression argEx = funcC.getArgument(paramIdx);
                            if (argEx instanceof ConditionalExpression) {
                            	List<Expression> argListCallSite = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)argEx, proc);
                            	for (Expression argCallSite: argListCallSite) {
                                    performGenKillSetting(cfgNode, defMapEntry, argCallSite, currentIR, proc, GenOption.GENONLY);
                            	}
                            } else {
                            	Expression refID = ChainTools.getRefIDExpression(argEx, proc);
                            	if (refID != null) {
                            		performGenKillSetting(cfgNode, defMapEntry, refID, currentIR, proc, GenOption.GENKILL);
                            	}
                            }
                        }
                    }
                }
//                if (ThirdPartyLibrary.hasSideEffectOnParameter(funcC)) {
//                    int[] paramIdxSet = ThirdPartyLibrary.getSideEffectParamIndices(funcC);
//                    if (paramIdxSet == null) {
//                        throw new RuntimeException("Side Effect Para Indices should be set for " + funcC.getName());
//                    }
//                    for (int paramIdx : paramIdxSet) {
//                        Expression argEx = funcC.getArgument(paramIdx);
//                        if (argEx instanceof UnaryExpression) {
//                            argEx = ((UnaryExpression) argEx).getExpression();
//                        }
//                        Expression pEx = ChainTools.getVarIDInMixedForm(argEx, proc);
//                        if (pEx != null) {
//                            argEx = pEx;
//                        }
//                        performGenKillSetting(cfgNode, defMapEntry, argEx, currentIR, proc);
//                    }
//                } else if (StandardLibrary.hasSideEffectOnParameter(funcC)) {
//                    int[] paramIdxSet = StandardLibrary.getSideEffectParamIndices(funcC);
//                    if (paramIdxSet == null) {
//                        throw new RuntimeException("Side Effect Para Indices should be set for " + funcC.getName());
//                    }
//                    for (int paramIdx : paramIdxSet) {
//                        if (paramIdx < funcC.getNumArguments()) {
//                            Expression argEx = funcC.getArgument(paramIdx);
//                            if (ChainTools.countIDEx(argEx) > 1) {
//                                List<Expression> defList = ChainTools.getUseList(argEx);
//                                for (Expression def : defList) {
//                                    if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                        performGenKillSetting(cfgNode, defMapEntry, def, currentIR, proc);
//                                    }
//                                }
//                            } else {
//                                performGenKillSetting(cfgNode, defMapEntry, argEx, currentIR, proc);
//                            }
//                        }
//                    }
//                }
            }
        }
    }

    private void extractKillSetFromSideEffectFunctions(
            Traversable traversableStmt,
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            Object currentIR,
            Procedure proc) {
        if (IRTools.containsFunctionCall(traversableStmt) == false) {
            return;
        }
        Set<AnalysisTarget> killSet = new HashSet();
        // def for ref before call
        Set<DFANode> callNodeSet = cfgNode.getData("psg_call_ref");
        if (callNodeSet != null) {
            for (DFANode callNode : callNodeSet) {
                Set<AnalysisTarget> outDEF = callNode.getData("OUTdef");
                for (AnalysisTarget target : outDEF) {
                    if (target.getProcedure().equals(proc)) {
                        killSet.add(target);
                    }
                }
            }
        }
        // def for global before call
        AnalysisTarget[] globalDefArray = new AnalysisTarget[globalDefList.size()];
        globalDefList.toArray(globalDefArray);
        Set<DFANode> callNodeGlobalSet = cfgNode.getData("psg_call_global");
        if (callNodeGlobalSet != null) {
            for (DFANode callNodeGlobal : callNodeGlobalSet) {
                BitSet bitOutDEF = callNodeGlobal.getData("OUTdef");
                for (int i = 0; i < globalDefArray.length; i++) {
                    if (bitOutDEF.get(i)) {
                        if (globalDefArray[i].getProcedure().equals(proc)) {
                            killSet.add(globalDefArray[i]);
                        }
                    }
                }
            }
        }
        // def for ref after call(on return)
        Set<DFANode> returnNodeSet = cfgNode.getData("psg_return_ref");
        if (returnNodeSet != null) {
            for (DFANode returnNode : returnNodeSet) {
                Set<AnalysisTarget> outDef = returnNode.getData("OUTdef");
                for (AnalysisTarget target : outDef) {
                    if (target.getProcedure().equals(proc)) {
                        killSet.remove(target);
                    }
                }
            }
        }
        // def for global after call(on return)
        Set<DFANode> returnNodeGlobalSet = cfgNode.getData("psg_return_global");
        if (returnNodeGlobalSet != null) {
            for (DFANode returnNodeGlobal : returnNodeGlobalSet) {
                BitSet bitOutDEF = returnNodeGlobal.getData("OUTdef");
                for (int i = 0; i < globalDefArray.length; i++) {
                    if (bitOutDEF.get(i)) {
                        if (globalDefArray[i].getProcedure().equals(proc)) {
                            killSet.remove(globalDefArray[i]);
                        }
                    }
                }
            }
        }
        //
        BitSet killBitSet = cfgNode.getData("KillSet");
        for (int i = 0; i < defMapEntry.length; i++) {
            for (AnalysisTarget target : killSet) {
                if (defMapEntry[i].equals(target)) {
                    killBitSet.set(i);
                }
            }
        }
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
