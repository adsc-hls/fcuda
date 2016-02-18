package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.hir.ConditionalExpression;
import cetus.hir.DepthFirstIterator;
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
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclarator;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class performs the reaching definition analysis for the program summary graph. * 
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class PSGReachingDefinitionAnalysis extends DataFlowAnalysis {

    private boolean summaryGraph = false;
    //
    private Map<Procedure, Set<AnalysisTarget>> refParamMap;
    private Map<AnalysisTarget, Integer> refParamIdxMap;
    //
    private boolean handleGlobal = false;
    private Set<Symbol> globalSet;
    private Set<AnalysisTarget> globalDefList;
    private Set<Procedure> globalDefProcSet;

    public PSGReachingDefinitionAnalysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            Map<Procedure, Set<AnalysisTarget>> refParamMap,
            Map<AnalysisTarget, Integer> refParamIdxMap) {
        super(program, cfgMap, true, true);
        this.analysisTargetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        this.summaryGraph = true;
        this.refParamMap = refParamMap;
        this.refParamIdxMap = refParamIdxMap;
    }

    public PSGReachingDefinitionAnalysis(
            Program program,
            Map<Procedure, CFGraph> cfgMap,
            Set<Symbol> globalSet,
            Set<AnalysisTarget> globalDefList,
            Set<Procedure> globalDefProcSet) {
        super(program, cfgMap, true, true);
        this.analysisTargetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        this.globalSet = globalSet;
        this.globalDefList = globalDefList;
        this.globalDefProcSet = globalDefProcSet;
        this.handleGlobal = true;
    }

    @Override
    public String getPassName() {
        return "[PSG-REACHING-DEFINITION-ANALYSIS]";
    }

    Set<AnalysisTarget> createAnalysisTargets(Procedure proc) {
        Set<AnalysisTarget> listDefMapping = new LinkedHashSet<AnalysisTarget>();
        if (handleGlobal) {
            for (AnalysisTarget defEx : globalDefList) {
                listDefMapping.add(new AnalysisTarget(defEx.getExpression(), defEx.getDFANode(), proc));
            }
            return listDefMapping;
        }
        Iterator<DFANode> cfgIter = cfgMap.get(proc).iterator();
        while (cfgIter.hasNext()) {
            DFANode cfgNode = cfgIter.next();
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
                    IDExpression idEx = ChainTools.getIDExpression(unaryEx);
                    if (handleGlobal) {
                        Symbol s = SymbolTools.getSymbolOf(idEx);
                        if (globalSet.contains(s)) {
                            listDefMapping.add(new AnalysisTarget(idEx, cfgNode, proc));
                        }
                    } else {
                        listDefMapping.add(new AnalysisTarget(idEx, cfgNode, proc));
                    }
                }
            } else if (currentIR instanceof VariableDeclarator) {
                if (handleGlobal) {
                    if (cfgNode.getData("param") != null) {
                        // handled by other statements
                    } else {
                        List<Expression> defSet = ChainTools.getDefListInDec((VariableDeclarator) currentIR);
                        for (Expression definedExpression : defSet) {
                            Symbol s = SymbolTools.getSymbolOf(definedExpression);
                            if (globalSet.contains(s)) {
                                listDefMapping.add(new AnalysisTarget(definedExpression, cfgNode, proc));
                            }
                        }
                    }
                } else {
                    if (cfgNode.getData("param") != null) {
                        listDefMapping.add(new AnalysisTarget(((VariableDeclarator) cfgNode.getData("ir")).getID(), cfgNode, proc));
                    } else {
                        List<Expression> defSet = ChainTools.getDefListInDec((VariableDeclarator) currentIR);
                        for (Expression definedExpression : defSet) {
                            listDefMapping.add(new AnalysisTarget(definedExpression, cfgNode, proc));
                        }
                    }
                }
            } else if (currentIR instanceof NestedDeclarator) {
                if (handleGlobal) {
                    if (cfgNode.getData("param") != null) {
                        // handled by other statements
                    } else {
                        List<Expression> defSet = ChainTools.getDefListInNestedDec((NestedDeclarator) currentIR);
                        for (Expression definedExpression : defSet) {
                            Symbol s = SymbolTools.getSymbolOf(definedExpression);
                            if (globalSet.contains(s)) {
                                listDefMapping.add(new AnalysisTarget(definedExpression, cfgNode, proc));
                            }
                        }
                    }
                } else {
                    if (cfgNode.getData("param") != null) {
                        listDefMapping.add(new AnalysisTarget(((NestedDeclarator) cfgNode.getData("ir")).getID(), cfgNode, proc));
                    } else {
                        List<Expression> defSet = ChainTools.getDefListInNestedDec((NestedDeclarator) currentIR);
                        for (Expression definedExpression : defSet) {
                            listDefMapping.add(new AnalysisTarget(definedExpression, cfgNode, proc));
                        }
                    }
                }
            } else if (currentIR instanceof SwitchStatement) {
                // Nothint to do on SwitchStatement
            } else if (currentIR instanceof Traversable) {
                if (handleGlobal) {
                    Traversable traversableStmt = (Traversable) currentIR;
//                    Set<Expression> defSet = DataFlowTools.getDefSet(traversableStmt);
                    List<Expression> defList = ChainTools.getPlainDefList(traversableStmt, proc);
                    // Add var to the def mapping table if it has the form of "var = ..."
                    for (Expression myEx : defList) {
                        Symbol s = SymbolTools.getSymbolOf(myEx);
                        if (globalSet.contains(s)) {
                            listDefMapping.add(new AnalysisTarget(myEx, cfgNode, proc));
                        }
                    }
                } else {
                    Traversable traversableStmt = (Traversable) currentIR;
//                    System.out.println("[createAnalysisTargets]currentIR: " + currentIR);
                    addPlainDefSetToDefMappingTable(listDefMapping, traversableStmt, cfgNode, proc);
                    if (IRTools.containsFunctionCall(traversableStmt)) {
                        addSideEffectStdLibParameterToDefMappingTable(listDefMapping, traversableStmt, cfgNode, proc);
                        if (summaryGraph) {
                            // each actual reference parameter at a call site
                            // is "use" followed by "def" that kills
                            List<FunctionCall> fcList = IRTools.getFunctionCalls(traversableStmt);
                            for (FunctionCall fc : fcList) {
                                Procedure callee = fc.getProcedure();
                                if (callee == null) {
                                    continue;
                                }
                                if (refParamMap.containsKey(callee)) {
                                    Set<AnalysisTarget> refList = refParamMap.get(callee);
                                    for (AnalysisTarget at : refList) {
                                    	Expression argCallSite = fc.getArgument(refParamIdxMap.get(at).intValue());
                                    	if (argCallSite instanceof ConditionalExpression) {
                                    		List<Expression> argListCallSite = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)argCallSite, proc);
                                    		for (Expression arg: argListCallSite) {
												AnalysisTarget newTarget = new AnalysisTarget(arg, cfgNode, proc);
												newTarget.setDummy();
												listDefMapping.add(newTarget);
                                    		}
                                    	} else {
                                    		Expression arg = ChainTools.getRefIDExpression(argCallSite, proc);
	                                        if (arg != null) {
												AnalysisTarget newTarget = new AnalysisTarget(arg, cfgNode, proc);
												newTarget.setDummy();
												listDefMapping.add(newTarget);
	                                        }
                                    	}
                                    }
                                }
                            }
                        }
                    }
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
        Iterator<DFANode> cfgIter = cfgMap.get(proc).iterator();
        while (cfgIter.hasNext()) {
            DFANode cfgNode = cfgIter.next();
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
                    extractGenKillSetFromSideEffectFunctions(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
                }
            } else if (currentIR instanceof VariableDeclarator) {
                if (cfgNode.getData("param") != null) {
                    performGenKillSetting(cfgNode, defMapEntry, ((VariableDeclarator) cfgNode.getData("ir")).getID(), currentIR, proc, GenOption.GENKILL);
                } else {
                    List<Expression> defSet = ChainTools.getDefListInDec((VariableDeclarator) currentIR);
                    for (Expression defEx : defSet) {
                        performGenKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc, GenOption.GENKILL);
                    }
                }
                extractGenKillSetForLibraryCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
                extractGenKillSetFromSideEffectFunctions((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
            } else if (currentIR instanceof NestedDeclarator) {
                if (cfgNode.getData("param") != null) {
                    performGenKillSetting(cfgNode, defMapEntry, ((NestedDeclarator) cfgNode.getData("ir")).getID(), currentIR, proc, GenOption.GENKILL);
                } else {
                    List<Expression> defSet = ChainTools.getDefListInNestedDec((NestedDeclarator) currentIR);
                    for (Expression defEx : defSet) {
                        performGenKillSetting(cfgNode, defMapEntry, defEx, currentIR, proc, GenOption.GENKILL);
                    }
                }
                extractGenKillSetForLibraryCalls((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
                extractGenKillSetFromSideEffectFunctions((Traversable) currentIR, cfgNode, defMapEntry, currentIR, proc);
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
                extractGenKillSetFromSideEffectFunctions(traversableStmt, cfgNode, defMapEntry, currentIR, proc);
            } else {
                throw new RuntimeException("Unexpected Statement IR: " + currentIR.toString() + ", Proc: " + proc.getSymbolName() + ", Class: " + currentIR.getClass().getCanonicalName());
            }
        }
    }

    private void addPlainDefSetToDefMappingTable(Set<AnalysisTarget> listDefMapping, Traversable tr, DFANode node, Procedure proc) {
        List<Expression> defList = ChainTools.getPlainDefList(tr, proc); //DataFlowTools.getDefList(tr);
        // Add var to the def mapping table if it has the form of "var = ..."
        for (Expression myEx : defList) {
//            System.out.println("[addPlainDefSetToDefMappingTable]def: " + myEx + ", class: " + myEx.getClass().getCanonicalName());
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
                        }
                    }
                }
            }
        }
    }

    private void printDefinitionMappingTable(Set<AnalysisTarget> defList, Procedure proc) {
        System.out.println("Def Mapping Table for : " + proc.getSymbolName());
        int idx = 0;
        for (AnalysisTarget def : defList) {
            System.out.println("DEF [" + idx++ + "]= " + def.getExpression() + ", DFANode: " + CFGraph.getIR(def.getDFANode()));
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
        	expToProcess = ChainTools.getRefIDExpression(expression, proc);
        }
        ChainTools.setGenBit(cfgNode, defMapEntry, expToProcess);
        if (genOption == GenOption.GENKILL) {
        	// for conditional expression parameter
	        ChainTools.setKillBit(cfgNode, defMapEntry, expToProcess, proc);
	        if (currentIR instanceof Statement) {
	            ChainTools.setKillBitForAlias(cfgNode, defMapEntry, expToProcess, (Statement) currentIR, proc);
	        }
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
                            Expression arg = funcC.getArgument(paramIdx);
                            if (arg instanceof ConditionalExpression) {
                            	List<Expression> argListCallSite = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)arg, proc);
                            	for (Expression argCallSite: argListCallSite) {
                                    performGenKillSetting(cfgNode, defMapEntry, argCallSite, currentIR, proc, GenOption.GENONLY);
                            	}
                            } else {
                            	Expression refID = ChainTools.getRefIDExpression(arg, proc);
                            	if (refID != null) {
                            		performGenKillSetting(cfgNode, defMapEntry, refID, currentIR, proc, GenOption.GENKILL);
                            	}
                            }
                        }
                    }
                }
            }
        }
    }

    private void extractGenKillSetFromSideEffectFunctions(
            Traversable traversableStmt,
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            Object currentIR,
            Procedure proc) {
        if (IRTools.containsFunctionCall(traversableStmt)) {
            Iterator stmtIter = new DepthFirstIterator(traversableStmt);
            while (stmtIter.hasNext()) {
                Object stmtIR = stmtIter.next();
                if (stmtIR instanceof FunctionCall) {
                    FunctionCall funcCall = (FunctionCall) stmtIR;
                    Procedure callee = funcCall.getProcedure();
                    if (callee != null) {
                        if (summaryGraph) {
                            // for subgraph, kill all the def when it meets a function call with ref
                            if (refParamMap.containsKey(callee)) {
                                // GenSet
                                BitSet genBitSet = cfgNode.getData("GenSet");
                                BitSet killBitSet = cfgNode.getData("KillSet");
                                Set<AnalysisTarget> refSet = refParamMap.get(callee);
                                for (AnalysisTarget at : refSet) {
                                    Expression arg = funcCall.getArgument(refParamIdxMap.get(at).intValue());
                                    if (arg instanceof ConditionalExpression) {
                                    	List<Expression> argListCallSite = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)arg, proc);
                                    	for (Expression argCallSite: argListCallSite) {
                                            for (int i = 0; i < defMapEntry.length; i++) {
                                                performGenKillSetting(cfgNode, defMapEntry, argCallSite, currentIR, proc, GenOption.GENONLY);
                                            }
                                    	}
                                    } else {
                                    	Expression refID = ChainTools.getRefIDExpression(arg, proc);
                                    	if (refID != null) {
                                    		performGenKillSetting(cfgNode, defMapEntry, refID, currentIR, proc, GenOption.GENKILL);
                                    	}
                                    }
                                }
                            }
                        } else if (handleGlobal) {
                            if (globalDefProcSet.contains(callee)) {
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
	            dfaNode.removeData("OutSet");
	            dfaNode.removeData("KillSet");
	            dfaNode.removeData("GenSet");
	        }
		} else {
	        Iterator cfgIter = cfgMap.get(targetProc).iterator();
	        while (cfgIter.hasNext()) {
	            DFANode dfaNode = (DFANode) cfgIter.next();
	            dfaNode.removeData("OutSet");
//	            dfaNode.removeData("KillSet");
//	            dfaNode.removeData("GenSet");
	        }
		}
	}

}
