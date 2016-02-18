package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.hir.BinaryExpression;
import cetus.hir.ConditionalExpression;
import cetus.hir.DataFlowTools;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.NestedDeclarator;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.StandardLibrary;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclarator;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class performs the upwards exposed uses analysis within the program summary graph.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class PSGUpwardsExposedUsesAnalysis extends DataFlowAnalysis {

    private boolean summaryGraph = false;
    //
    private Map<Procedure, Set<AnalysisTarget>> refParamMap;
    private Map<AnalysisTarget, Integer> refParamIdxMap;
    //
    private boolean handleGlobal = false;
    private Set<Symbol> globalSet;
    private Set<AnalysisTarget> globalUseList;
    private Set<Procedure> globalUseProcSet;

    public PSGUpwardsExposedUsesAnalysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            Map<Procedure, Set<AnalysisTarget>> refParamMap,
            Map<AnalysisTarget, Integer> refParamIdxMap) {
        super(program, cfgMap, false, true);
        this.analysisTargetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        this.summaryGraph = true;
        this.refParamMap = refParamMap;
        this.refParamIdxMap = refParamIdxMap;
    }

    public PSGUpwardsExposedUsesAnalysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            Set<Symbol> globalSet,
            Set<AnalysisTarget> globalUseList,
            Set<Procedure> globalUseProcSet) {
        super(program, cfgMap, false, true);
        this.analysisTargetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        this.globalSet = globalSet;
        this.globalUseList = globalUseList;
        this.globalUseProcSet = globalUseProcSet;
        this.handleGlobal = true;
    }

    @Override
    public String getPassName() {
        return "[PSG-UPWARDS-EXPOSED-USES-ANALYSIS]";
    }

    Set<AnalysisTarget> createAnalysisTargets(Procedure proc) {
        Set<AnalysisTarget> listUsesMapping = new LinkedHashSet<AnalysisTarget>();
        if (handleGlobal) {
            for (AnalysisTarget useEx : globalUseList) {
                listUsesMapping.add(new AnalysisTarget(useEx.getExpression(), useEx.getDFANode(), proc));
            }
//            printUseMappingTable(listUsesMapping, proc);
            return listUsesMapping;
        }
        Iterator cfgIter = cfgMap.get(proc).iterator();
        while (cfgIter.hasNext()) {
            DFANode cfgNode = (DFANode) cfgIter.next();
            Object currentIR = CFGraph.getIR(cfgNode);
            if (currentIR == null) {
                // The first cfgNode in CFG is always null. Skip this cfgNode.
                continue;
            }
            if (currentIR instanceof VariableDeclarator) {
                if (handleGlobal) {
                    if (cfgNode.getData("param") != null) {
                        // handled by other statements
                    } else {
                        Traversable traversableStmt = (Traversable) currentIR;
                        List<Expression> useList = ChainTools.getUseList(traversableStmt);
                        for (Expression myEx : useList) {
                            if (ChainTools.getIDExpression(myEx) != null) {
                                Symbol s = SymbolTools.getSymbolOf(myEx);
                                if (globalSet.contains(s)) {
                                    listUsesMapping.add(new AnalysisTarget(myEx, cfgNode, proc));
                                }
                            }
                        }
                    }
                } else {
                    if (cfgNode.getData("param") != null) {
                        // do nothing, param is only def
                    } else {
                        Traversable traversableStmt = (Traversable) currentIR;
                        List<Expression> useList = ChainTools.getUseList(traversableStmt);
                        for (Expression myEx : useList) {
                            if (ChainTools.getIDExpression(myEx) != null) {
                                listUsesMapping.add(new AnalysisTarget(myEx, cfgNode, proc));
                            }
                        }
                    }
                }
            } else if (currentIR instanceof NestedDeclarator) {
                if (handleGlobal) {
                    if (cfgNode.getData("param") != null) {
                        // handled by other statements
                    } else {
                        Traversable traversableStmt = (Traversable) currentIR;
                        List<Expression> useList = ChainTools.getUseList(traversableStmt);
                        for (Expression myEx : useList) {
                            if (ChainTools.getIDExpression(myEx) != null) {
                                Symbol s = SymbolTools.getSymbolOf(myEx);
                                if (globalSet.contains(s)) {
                                    listUsesMapping.add(new AnalysisTarget(myEx, cfgNode, proc));
                                }
                            }
                        }
                    }
                } else {
                    if (cfgNode.getData("param") != null) {
                        // do nothing, param is only def
                    } else {
                        Traversable traversableStmt = (Traversable) currentIR;
                        List<Expression> useList = ChainTools.getUseList(traversableStmt);
                        for (Expression myEx : useList) {
                            if (ChainTools.getIDExpression(myEx) != null) {
                                listUsesMapping.add(new AnalysisTarget(myEx, cfgNode, proc));
                            }
                        }
                    }
                }
            } else if (currentIR instanceof SwitchStatement) {
                // Nothint to do on SwitchStatement
            } else if (currentIR instanceof Traversable) {
                if (handleGlobal) {
                    Traversable traversableStmt = (Traversable) currentIR;
                    List<Expression> useList = ChainTools.getUseList(traversableStmt);
                    for (Expression myEx : useList) {
                        if (ChainTools.getIDExpression(myEx) != null) {
                            Symbol s = SymbolTools.getSymbolOf(myEx);
                            if (globalSet.contains(s)) {
                                listUsesMapping.add(new AnalysisTarget(myEx, cfgNode, proc));
                            }
                        }
                    }
                } else {
                    Traversable traversableStmt = (Traversable) currentIR;
                    List<Expression> useList = ChainTools.getUseList(traversableStmt);
//                    System.out.println("tarversableStmt: " + traversableStmt);
                    for (Expression myEx : useList) {
//                        System.out.println("myEx: " + myEx + ", class: " + myEx.getClass().getCanonicalName());
                        Expression exID = ChainTools.getIDExpression(myEx);
                        if (exID != null) {
                            listUsesMapping.add(new AnalysisTarget(myEx, cfgNode, proc));
                        }
                    }
                }
            } else {
                throw new RuntimeException("Unexpected Statement: IR: " + currentIR.toString() + ", Proc: " + proc.getSymbolName() + ", Class: " + currentIR.getClass().getCanonicalName());
            }
        }
//        printUseMappingTable(listUsesMapping, proc);
        return listUsesMapping;
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
            // extract GEN set
            if (currentIR instanceof Traversable) {
                Traversable traversableStmt = (Traversable) currentIR;
                List<Expression> useList = ChainTools.getUseList(traversableStmt);
                for (Expression useEx : useList) {
                    if (ChainTools.getIDExpression(useEx) != null) {
                        performGenSetting(cfgNode, defMapEntry, useEx, currentIR, proc);
                    }
                }
            }
            // extract KILL set
            if (currentIR instanceof UnaryExpression) {
                UnaryExpression unaryEx = (UnaryExpression) currentIR;
                UnaryOperator unaryOp = unaryEx.getOperator();
                if (unaryOp == UnaryOperator.POST_DECREMENT ||
                        unaryOp == UnaryOperator.POST_INCREMENT ||
                        unaryOp == UnaryOperator.PRE_DECREMENT ||
                        unaryOp == UnaryOperator.PRE_INCREMENT) {
                    Expression myEx = unaryEx.getExpression();
                    performKillSetting(cfgNode, defMapEntry, myEx, currentIR, proc);
                } else {
                    Traversable traversableStmt = (Traversable) currentIR;
//                    Set<Expression> defSet = DataFlowTools.getDefSet(traversableStmt);
                    List<Expression> defList = ChainTools.getPlainDefList(traversableStmt, proc);
                    // Set KILL for Def Set for the procedure
                    // handle the case of assignment "var = xxx"
                    for (Expression defEx : defList) {
                        performKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc);
                    }
                    extractKillSetForLibraryCalls(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
                    extractKillSetWithFunctionCalls(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
                }
            } else if (currentIR instanceof VariableDeclarator) {
                if (cfgNode.getData("param") != null) {
                    performKillSetting(cfgNode, defMapEntry, ((VariableDeclarator) cfgNode.getData("ir")).getID(), currentIR, proc);
                } else {
                    List<Expression> defSet = ChainTools.getDefListInDec((VariableDeclarator) currentIR);
                    for (Expression defEx : defSet) {
                        performKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc);
                    }
                }
                extractKillSetForLibraryCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
                extractKillSetWithFunctionCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
            } else if (currentIR instanceof NestedDeclarator) {
                if (cfgNode.getData("param") != null) {
                    performKillSetting(cfgNode, defMapEntry, ((NestedDeclarator) cfgNode.getData("ir")).getID(), currentIR, proc);
                } else {
                    List<Expression> defSet = ChainTools.getDefListInNestedDec((NestedDeclarator) currentIR);
                    for (Expression defEx : defSet) {
                        performKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc);
                    }
                }
                extractKillSetForLibraryCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
                extractKillSetWithFunctionCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
            } else if (currentIR instanceof SwitchStatement) {
                // nothing to do on SwitchStatement
            } else if (currentIR instanceof Traversable) {
                Traversable traversableStmt = (Traversable) currentIR;
//                Set<Expression> defSet = DataFlowTools.getDefSet(traversableStmt);
                List<Expression> defList = ChainTools.getPlainDefList(traversableStmt, proc);
                // Set GEN and KILL for Def Set for the procedure
                // handle the case of assignment "var = xxx"
                for (Expression defEx : defList) {
                    performKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc);
                }
                extractKillSetForLibraryCalls(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
                extractKillSetWithFunctionCalls(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
            } else {
                throw new RuntimeException("Unexpected Statement IR: " + currentIR.toString() + ", Proc: " + proc.getSymbolName() + ", Class: " + currentIR.getClass().getCanonicalName());
            }
        }
    }

    private void printUseMappingTable(Set<AnalysisTarget> useList, Procedure proc) {
        System.out.println("Use Mapping Table for : " + proc.getSymbolName());
        int idx = 0;
        for (AnalysisTarget use : useList) {
            if (handleGlobal) {
                System.out.println("USE[" + idx++ + "] = " + use.getExpression());
            } else {
                System.out.println("USE[" + idx++ + "] = " + use.getExpression() + ", DFANode: " + CFGraph.getIR(use.getDFANode()));
            }
        }
    }

    private void performKillSetting(
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            Expression expression,
            Object currentIR,
            Procedure proc) {
        Expression expToProcess = null;
        if (ChainTools.isArrayAccessWithConstantIndex(expression)) {
            expToProcess = expression;
        } else if (ChainTools.isStructureAccess(expression, proc)) {
            expToProcess = ChainTools.getIDVariablePlusMemberInStruct(expression);
        } else {
            expToProcess = ChainTools.getRefIDExpression(expression, proc);//getIDExpression(expression);
        }
        ChainTools.setKillBit(cfgNode, defMapEntry, expToProcess, proc);
        if (currentIR instanceof Statement) {
            ChainTools.setKillBitForAlias(cfgNode, defMapEntry, expToProcess, (Statement) currentIR, proc);
        }
    }

    private void performGenSetting(
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            Expression expression,
            Object currentIR,
            Procedure proc) {
        Expression expToProcess = expression;
        if (expression instanceof UnaryExpression) {
            if (SymbolTools.isPointer(expression)) {
                Expression idEx = ChainTools.getIDExpression(expression);
                if (idEx instanceof Identifier) {
                    if (SymbolTools.isPointer(((Identifier) idEx).getSymbol()) == false) {
                        // Accessing address but not declared as pointer
                        expToProcess = ((UnaryExpression) expression).getExpression();
                    }
                } else {
                    throw new RuntimeException("Identifier expected but another is returned: expression: " + expression + ", idEx: " + idEx + ", expression class: " + expression.getClass().getCanonicalName());
                }
            }
        }
        if (ChainTools.isArrayAccess(expression)) {
            if (ChainTools.isArrayAccessWithConstantIndex(expression) == false) {
                expToProcess = ChainTools.getIDExpression(expression);
            }
        } else if (ChainTools.isStructureAccess(expression, proc)) {
            expToProcess = ChainTools.getIDVariablePlusMemberInStruct(expression);
        }
        ChainTools.setGenBit(cfgNode, defMapEntry, expToProcess);
    }

    private void extractKillSetForLibraryCalls(
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
//                            	List<Expression> argListCallSite = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)argEx, proc);
//                            	for (Expression argCallSite: argListCallSite) {
//                                    performKillSetting(cfgNode, defMapEntry, argCallSite, currentIR, proc);
//                            	}
                            } else {
                            	Expression refID = ChainTools.getRefIDExpression(argEx, proc);
                            	if (refID != null) {
                            		performKillSetting(cfgNode, defMapEntry, refID, currentIR, proc);
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
//                        performKillSetting(cfgNode, defMapEntry, argEx, currentIR, proc);
//                    }
//                } else if (StandardLibrary.hasSideEffectOnParameter(funcC)) {
//                    int[] paramIdxSet = StandardLibrary.getSideEffectParamIndices(funcC);
//                    if (paramIdxSet == null) {
//                        throw new RuntimeException("Side Effect Para Indices should be set for " + funcC.getName());
//                    }
//                    for (int paramIdx : paramIdxSet) {
//                        if (paramIdx < funcC.getNumArguments()) {
//                            Expression argEx = funcC.getArgument(paramIdx);
//                            System.out.println("[extractKillSetForLibraryCalls]argEx: " + argEx + ", class: " + argEx.getClass().getCanonicalName());
//                            if (argEx instanceof BinaryExpression) {
//                                System.out.println("[extractKillSetForLibraryCalls]BinaryExpression: " + argEx);
//                                List<Expression> defList = ChainTools.getUseList(argEx);
//                                for (Expression def : defList) {
//                                    if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                        Expression idEx = ChainTools.getIDExpression(def);
//                                        if (idEx != null) {
//                                            performKillSetting(cfgNode, defMapEntry, idEx, currentIR, proc);
//                                        }
//                                    }
//                                }
//                            } else {
//                                performKillSetting(cfgNode, defMapEntry, argEx, currentIR, proc);
//                            }
//                        }
//                    }
//                }
            }
        }
    }

    private void extractKillSetWithFunctionCalls(
            Traversable traversableStmt,
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            Object currentIR,
            Procedure proc) {
        if (IRTools.containsFunctionCall(traversableStmt)) {
            DepthFirstIterator stmtIter = new DepthFirstIterator(traversableStmt);
            while (stmtIter.hasNext()) {
                Object stmtIR = stmtIter.next();
                if (stmtIR instanceof FunctionCall) {
                    FunctionCall funcCall = (FunctionCall) stmtIR;
                    Procedure callee = funcCall.getProcedure();
                    if (callee != null) {
                        if (summaryGraph) {
                            // for subgraph, kill all the def when it meets a function call with ref
                            if (refParamMap.containsKey(callee)) {
                                Set<AnalysisTarget> refSet = refParamMap.get(callee);
                                BitSet killBitSet = cfgNode.getData("KillSet");
                                for (AnalysisTarget at : refSet) {
                                    Expression arg = funcCall.getArgument(refParamIdxMap.get(at).intValue());
                                    if (arg instanceof ConditionalExpression) {
//                                    	List<Expression> argListCallSite = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)arg, proc);
//                                    	for (Expression argCallSite: argListCallSite) {
//                                            for (int i = 0; i < defMapEntry.length; i++) {
//                                                if (ChainTools.getIDExpression(defMapEntry[i].getExpression()).equals(argCallSite)) {
//                                                    killBitSet.set(i);
//                                                }
//                                            }
//                                    	}
//                                        Expression trueEx = ((ConditionalExpression) arg).getTrueExpression();
//                                        if (trueEx instanceof BinaryExpression) {
//                                            List<Expression> defList = ChainTools.getUseList(trueEx);
//                                            for (Expression def : defList) {
//                                                if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                                    IDExpression idEx = ChainTools.getIDExpression(def);
//                                                    if (idEx != null) {
//                                                        for (int i = 0; i < defMapEntry.length; i++) {
//                                                            if (ChainTools.getIDExpression(defMapEntry[i].getExpression()).equals(idEx)) {
//                                                                killBitSet.set(i);
//                                                            }
//                                                        }
//                                                    }
//                                                }
//                                            }
//                                        } else {
//                                            Expression idEx = ChainTools.getIDExpression(trueEx);
//                                            if (idEx != null) {
//                                                for (int i = 0; i < defMapEntry.length; i++) {
//                                                    if (ChainTools.getIDExpression(defMapEntry[i].getExpression()).equals(idEx)) {
//                                                        killBitSet.set(i);
//                                                    }
//                                                }
//                                            }
//                                        }
//                                        Expression falseEx = ((ConditionalExpression) arg).getTrueExpression();
//                                        if (falseEx instanceof BinaryExpression) {
//                                            List<Expression> defList = ChainTools.getUseList(falseEx);
//                                            for (Expression def : defList) {
//                                                if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                                    IDExpression idEx = ChainTools.getIDExpression(def);
//                                                    if (idEx != null) {
//                                                        for (int i = 0; i < defMapEntry.length; i++) {
//                                                            if (ChainTools.getIDExpression(defMapEntry[i].getExpression()).equals(idEx)) {
//                                                                killBitSet.set(i);
//                                                            }
//                                                        }
//                                                    }
//                                                }
//                                            }
//                                        } else {
//                                            Expression idEx = ChainTools.getIDExpression(falseEx);
//                                            if (idEx != null) {
//                                                for (int i = 0; i < defMapEntry.length; i++) {
//                                                    if (ChainTools.getIDExpression(defMapEntry[i].getExpression()).equals(idEx)) {
//                                                        killBitSet.set(i);
//                                                    }
//                                                }
//                                            }
//                                        }
//                                    } else if (arg instanceof BinaryExpression) {
//                                        List<Expression> defList = ChainTools.getUseList(arg);
//                                        for (Expression def : defList) {
//                                            if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                                IDExpression idEx = ChainTools.getIDExpression(def);
//                                                if (idEx != null) {
//                                                    for (int i = 0; i < defMapEntry.length; i++) {
//                                                        if (ChainTools.getIDExpression(defMapEntry[i].getExpression()).equals(idEx)) {
//                                                            killBitSet.set(i);
//                                                        }
//                                                    }
//                                                }
//                                            }
//                                        }
                                    } else {
                                        Expression idEx = ChainTools.getRefIDExpression(arg, proc);
                                        if (idEx != null) {
                                            for (int i = 0; i < defMapEntry.length; i++) {
                                                if (ChainTools.getIDExpression(defMapEntry[i].getExpression()).equals(idEx)) {
                                                    killBitSet.set(i);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else if (handleGlobal) {
                            if (globalUseProcSet.contains(callee)) {
                                BitSet killBitSet = cfgNode.getData("KillSet");
                                for (int i = 0; i < defMapEntry.length; i++) {
                                    killBitSet.set(i);
                                }
                                cfgNode.putData("KillSet", killBitSet);
                            }
                        } else {
                            throw new RuntimeException("Running Mode is not set properly. It should be either summary graph or handleGlobal.");
                        }
                    }
                }
            }
        }
    }

	@Override
	void cleanupUnnecessaryData() {
		if (handleGlobal == false) {
	        // delete unnecessary data
	        Iterator cfgIter = cfgMap.get(targetProc).iterator();
	        while (cfgIter.hasNext()) {
	            DFANode dfaNode = (DFANode) cfgIter.next();
	            dfaNode.removeData("InSet");
	            dfaNode.removeData("KillSet");
	            dfaNode.removeData("GenSet");
	        }
		}
	}
}
