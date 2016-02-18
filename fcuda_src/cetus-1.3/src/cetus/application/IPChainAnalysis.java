package cetus.application;

import cetus.analysis.AnalysisPass;
import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.analysis.Domain;
import cetus.analysis.IPAnalysis;
import cetus.analysis.IPPointsToAnalysis;
import cetus.analysis.PointsToDomain;
import cetus.analysis.PointsToRel;
import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.ConditionalExpression;
import cetus.hir.Expression;
import cetus.hir.Identifier;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.NestedDeclarator;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.BitSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * This class performs the interprocedural Def-Use and Use-Def chain computation.
 * It uses the program summary graph which is implemented in {@link ProgramSummaryGraph}.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class IPChainAnalysis extends AnalysisPass implements DefUseChain, UseDefChain {

    private Map<Procedure, CFGraph> cfgMap;
    private Map<Procedure, Set<AnalysisTarget>> defTargetSetMap;
    private Map<Procedure, Set<AnalysisTarget>> useTargetSetMap;
    private final Set<Procedure> procList;
    private LinkedHashSet<AnalysisTarget> globalDefSet;
    private LinkedHashSet<AnalysisTarget> globalUseSet;
    private boolean isAnalysisDone = false;

    public IPChainAnalysis(Program program) {
        super(program);
        procList = ChainTools.getProcedureSet(program);
    }

    @Override
    public String getPassName() {
        return "[IP-CHAIN-ANALYSIS]";
    }

    @Override
    public void start() {
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("#### Performing Alias analysis...");
        performAliasAnalysis();
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("#### Generating Control Flow Graph...");
//        generateCFGraph();
        generateCFGraphWithThread();
        // Interprocedural Information
        generateProgramSummaryGraph();
        // Intraprocedural Analysis with Program Summary Information
//        performReachingDefinitionAnalysis();
        performReachingDefinitionAnalysisWithThread();
        // Perform UD & DU chain computation
        computeChains();
        isAnalysisDone = true;
    }

    private void performAliasAnalysis() {
        IPAnalysis analysis = new IPPointsToAnalysis(program);
        analysis.start();
    }

    private void generateCFGraph() {
        cfgMap = new HashMap<Procedure, CFGraph>();
        for (Procedure proc : procList) {
            List<VariableDeclaration> paramList = proc.getParameters();
        	if (PrintTools.getVerbosity() > 1)
        		System.out.println("[generateCFGraph]proc: " + proc.getSymbolName());
            cfgMap.put(proc, new CFGraph(proc, null, true, paramList));
        }
    }
    
    private void generateCFGraphWithThread() {
        cfgMap = new HashMap<Procedure, CFGraph>();
        ExecutorService taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
        	CFGBuilder builder = new CFGBuilder(cfgMap, proc);
        	taskExecutor.execute(builder);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion
    }

    class CFGBuilder implements Runnable {

    	private Map<Procedure, CFGraph> cfgMap;
    	private Procedure targetProc;
    	
    	CFGBuilder(Map<Procedure, CFGraph> cfgMap, Procedure targetProc) {
    		this.cfgMap = cfgMap;
    		this.targetProc = targetProc;
    	}
    	
		@Override
		public void run() {
	    	if (PrintTools.getVerbosity() > 1)
	    		System.out.println("Generating CFG for " + targetProc.getSymbolName() + " ...");
            List<VariableDeclaration> paramList = targetProc.getParameters();
			CFGraph cfg = new CFGraph(targetProc, null, true, paramList);
			synchronized (cfgMap) {
	            cfgMap.put(targetProc, cfg);
			}
		}    	
    }
    
    private void generateProgramSummaryGraph() {
        ProgramSummaryGraph psg = new ProgramSummaryGraph(program);
        psg.buildGraph(cfgMap);
        globalDefSet = (LinkedHashSet<AnalysisTarget>) psg.getGlobalDefSet();
        globalUseSet = (LinkedHashSet<AnalysisTarget>) psg.getGlobalUseSet();
    }

    private void performReachingDefinitionAnalysis() {
        ReachingDefinitionAnalysis rdAnalysis = new ReachingDefinitionAnalysis(program, cfgMap, globalDefSet);
        rdAnalysis.start();
        defTargetSetMap = rdAnalysis.getAnalysisTargetListMap();
        useTargetSetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
    }
    
    private void performReachingDefinitionAnalysisWithThread() {
        defTargetSetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        ExecutorService taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            ReachingDefinitionAnalysis rdAnalysis = new ReachingDefinitionAnalysis(program, cfgMap, globalDefSet);
            rdAnalysis.setAnalysisTarget(proc, defTargetSetMap);
        	taskExecutor.execute(rdAnalysis);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion
        useTargetSetMap = new HashMap<Procedure, Set<AnalysisTarget>>();
    }

    private void computeChains() {
        for (Procedure proc : procList) {
            compute(proc);
        }
    }

    private void addUseDefChain(AnalysisTarget def, Expression use, DFANode node, Procedure proc) {
        if (def.getDFANode().getData("param") != null) {
            if (def.getProcedure().equals(proc) == false) {
                return;
            }
        }
        Expression idEx = null;
        if (ChainTools.isArrayAccessWithConstantIndex(use)) {
            idEx = use;
        } else {
            idEx = ChainTools.getIDExpression(use);
        }
        Set<AnalysisTarget> useTargetSet = useTargetSetMap.get(proc);
        if (useTargetSet == null) {
            useTargetSet = new LinkedHashSet<AnalysisTarget>();
            useTargetSetMap.put(proc, useTargetSet);
        }
        boolean useFound = false;
        for (AnalysisTarget useTarget : useTargetSet) {
            if (System.identityHashCode(useTarget.getExpression()) == System.identityHashCode(idEx)) {
                useTarget.addDefChain(def);
                useFound = true;
            }
        }
        if (useFound == false) {
            AnalysisTarget useTarget = new AnalysisTarget(use, node, proc);
            useTarget.addDefChain(def);
            useTargetSet.add(useTarget);
        }
    }

    private AnalysisTarget[] getDefTargetArray(Procedure proc) {
        Set<AnalysisTarget> targetSet = defTargetSetMap.get(proc);
        AnalysisTarget defMapEntry[] = new AnalysisTarget[targetSet.size()];
        targetSet.toArray(defMapEntry);
        return defMapEntry;
    }

    private AnalysisTarget[] getUseTargetArray(Procedure proc) {
        Set<AnalysisTarget> targetSet = useTargetSetMap.get(proc);
        if (targetSet == null) {
            return null;
        }
        AnalysisTarget useTargetArray[] = new AnalysisTarget[targetSet.size()];
        targetSet.toArray(useTargetArray);
        return useTargetArray;
    }

    private void generateChainForDefinedArrayDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defSetArray,
            VariableDeclarator declarator,
            Procedure proc) {
        List<Expression> useList = ChainTools.getUseListInDec(declarator);
        for (Expression currentUse : useList) {
            boolean defFound = false;
            for (int i = 0; i < defSetArray.length; i++) {
                if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse) &&
                        ChainTools.isDefInItself(cfgNode, defSetArray[i].getExpression(), currentUse)) {
                    defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    defFound = true;
                    break;
                }
            }
            if (defFound == false) {
                for (int i = 0; i < defSetArray.length; i++) {
                    if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse) &&
                            ChainTools.checkInBitSet(cfgNode, i)) {
                        defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        }
    }

    private void generateChainForDefinedArrayNestedDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defSetArray,
            NestedDeclarator declarator,
            Procedure proc) {
        List<Expression> useList = ChainTools.getUseListInNestedDec(declarator);
        for (Expression currentUse : useList) {
            boolean defFound = false;
            for (int i = 0; i < defSetArray.length; i++) {
                if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse) &&
                        ChainTools.isNestedDefInItself(cfgNode, defSetArray[i].getExpression(), currentUse)) {
                    defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    defFound = true;
                    break;
                }
            }
            if (defFound == false) {
                for (int i = 0; i < defSetArray.length; i++) {
                    if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse) &&
                            ChainTools.checkInBitSet(cfgNode, i)) {
                        defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        }
    }

    private void generateChainForPlainDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            VariableDeclarator declarator,
            Procedure proc) {
        List<Expression> useSet = ChainTools.getUseListInDec(declarator);
        for (Expression currentUse : useSet) {
            for (int i = 0; i < defMapEntry.length; i++) {
                if (ChainTools.isArrayAccess(currentUse)) {
                    generateChainForArrayAccess(currentUse, defMapEntry, cfgNode, proc);
                } else if (ChainTools.isPointerAccess(currentUse)) {
                    if (ChainTools.isStructureAccess(currentUse, proc)) {
                        generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                    } else {
                        generateChainForPointerAccess(currentUse, defMapEntry, cfgNode, proc);
                    }
                } else if (ChainTools.isStructureAccess(currentUse, proc)) {
                    generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                } else {
                    generateChainForPlainExpression(currentUse, defMapEntry, cfgNode, proc);
                }
                Object nodeIR = CFGraph.getIR(cfgNode);
                generateChainForAlias(nodeIR, currentUse, defMapEntry, cfgNode, proc);
            }
        }
    }

    private void generateChainForPlainNestedDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            NestedDeclarator declarator,
            Procedure proc) {
        List<Expression> useSet = ChainTools.getUseListInNestedDec(declarator);
        for (Expression currentUse : useSet) {
            for (int i = 0; i < defMapEntry.length; i++) {
                if (ChainTools.isArrayAccess(currentUse)) {
                    generateChainForArrayAccess(currentUse, defMapEntry, cfgNode, proc);
                } else if (ChainTools.isPointerAccess(currentUse)) {
                    if (ChainTools.isStructureAccess(currentUse, proc)) {
                        generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                    } else {
                        generateChainForPointerAccess(currentUse, defMapEntry, cfgNode, proc);
                    }
                } else if (ChainTools.isStructureAccess(currentUse, proc)) {
                    generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                } else {
                    generateChainForPlainExpression(currentUse, defMapEntry, cfgNode, proc);
                }
                Object nodeIR = CFGraph.getIR(cfgNode);
                generateChainForAlias(nodeIR, currentUse, defMapEntry, cfgNode, proc);
            }
        }
    }

    private void generateChainForArrayAccess(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
        if (ChainTools.isArrayAccessWithNoIndex(currentUse) ||
                ChainTools.isArrayAccessWithVariableIndex(currentUse) ||
                ChainTools.isArrayAccessWithPartiallyConstantIndex(currentUse)) {
            for (int i = 0; i < defTargetArray.length; i++) {
                if (ChainTools.checkInBitSet(cfgNode, i) &&
                        ChainTools.hasSameArrayIdentifier(currentUse, defTargetArray[i].getExpression())) {
                    defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                }
            }
        } else if (ChainTools.isArrayAccessWithConstantIndex(currentUse)) {
            for (int i = 0; i < defTargetArray.length; i++) {
                if (ChainTools.checkInBitSet(cfgNode, i) &&
                        ChainTools.hasSameArrayIdentifier(currentUse, defTargetArray[i].getExpression())) {
                    if (ChainTools.isArrayAccessWithConstantIndex(defTargetArray[i].getExpression())) {
                        if (ChainTools.hasSameToString(currentUse, defTargetArray[i].getExpression())) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    } else {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        } else {
            ChainTools.traverseIR(currentUse);
            throw new RuntimeException("Unexpected Expression: " + "currentUse: " + currentUse + " (Unknown ArrayType)");
        }
    }

    private void generateChainForStructure(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
        for (int i = 0; i < defTargetArray.length; i++) {
            // handle exact matching
            if (ChainTools.isStructureAccess(defTargetArray[i].getExpression(), proc) &&
                    ChainTools.checkInBitSet(cfgNode, i)) {
                if (ChainTools.getMemberOnlyInStruct(currentUse) != null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) != null) {
                    // use: var->member, def: var->member case
                    if (ChainTools.isArrayAccessInStruct(currentUse, proc)) {
                        if (ChainTools.hasSameArrayAccessInStruct(defTargetArray[i].getExpression(), currentUse, proc)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    } else {
                        if (ChainTools.hasSameToString(
                                ChainTools.getIDVariablePlusMemberInStruct(defTargetArray[i].getExpression()),
                                ChainTools.getIDVariablePlusMemberInStruct(currentUse))) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    }
                } else if (ChainTools.getMemberOnlyInStruct(currentUse) == null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) == null) {
                    // use: var, def: var case
                    if (ChainTools.hasSameToString(defTargetArray[i].getExpression(), currentUse)) {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                } else if (ChainTools.getMemberOnlyInStruct(currentUse) != null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) == null) {
                    // use: var->member, def: var case
                    if (ChainTools.isArrayAccessInStruct(currentUse, proc)) {
                        if (ChainTools.hasSameArrayAccessInStruct(defTargetArray[i].getExpression(), currentUse, proc)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    } else {
                        if (ChainTools.hasSameStructureVariableIdentifier(defTargetArray[i].getExpression(), currentUse, proc)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    }
                } else if (ChainTools.getMemberOnlyInStruct(currentUse) == null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) != null) {
                    // use: var, def: var->member case
                    if (ChainTools.hasSameStructureVariableIdentifier(defTargetArray[i].getExpression(), currentUse, proc)) {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                } else {
                    throw new RuntimeException("currentUse: " + currentUse + ", defMapEntry[i].expression: " + defTargetArray[i].getExpression());
                }
            }
        } //  for (int i = 0; i < useTargetArray.length; i++)
    }

    private void generateChainForPlainExpression(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
        for (int i = 0; i < defTargetArray.length; i++) {
            // handle exact matching
            if (ChainTools.hasSameToString(defTargetArray[i].getExpression(), currentUse)) {
                // if bit for this defEx is set as 1
                if (ChainTools.checkInBitSet(cfgNode, i)) {
                    defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                }
            } else if (defTargetArray[i].getExpression() instanceof ArrayAccess ||
                    defTargetArray[i].getExpression() instanceof UnaryExpression) {
                // Handle the case of accessing array and pointer dereferencing
                Expression id = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                if (ChainTools.hasSameToString(id, currentUse)) {
                    if (ChainTools.checkInBitSet(cfgNode, i)) {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        } //  for (int i = 0; i < useTargetArray.length; i++)
    }

    private void generateChainForPointerAccess(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
        for (int i = 0; i < defTargetArray.length; i++) {
            // handle exact matching
            if (ChainTools.hasSameToString(defTargetArray[i].getExpression(), currentUse)) {
                // if bit for this defEx is set as 1
                if (ChainTools.checkInBitSet(cfgNode, i)) {
                    defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                }
            } else {
                // Handle the case of accessing array and pointer dereferencing
                Expression defId = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                Expression useId = ChainTools.getIDExpression(currentUse);
                if (defId == null || useId == null) {
                    continue;
                }
                if (currentUse instanceof ArrayAccess) {
                    if (ChainTools.hasSameToString(defId, useId)) {
                        if (ChainTools.checkInBitSet(cfgNode, i)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    }
                } else {
                    if (ChainTools.hasSameToString(defId, currentUse)) {
                        if (ChainTools.checkInBitSet(cfgNode, i)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    }
                }
            }
        } //  for (int i = 0; i < useTargetArray.length; i++)
    }

    private void generateChainForAlias(
            Object nodeIR,
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
        // Apply the alias analysis information to the UD chain
        // If the alias is definite, the def of the alias should be in the def chain
        // If the alias is possible, the def of itself and the alias should be in the def chain
        if (nodeIR instanceof Statement) {
            Set<Symbol> definedSymbolsInProc = ChainTools.getDefSymbol(defTargetSetMap.get(proc), proc);//DataFlowTools.getDefSymbol(proc);
            Expression cuId = ChainTools.getIDExpression(currentUse);
            if (cuId == null || (cuId instanceof Identifier) == false) {
                return;
            }
            Symbol symForCurrentEx = ((Identifier) cuId).getSymbol();
            Domain aliasInfo = IPPointsToAnalysis.getPointsToRelations((Statement) nodeIR);
            PointsToDomain aliasInfoForCurrentStmt = null;
            if (aliasInfo instanceof PointsToDomain) {
                aliasInfoForCurrentStmt = (PointsToDomain) aliasInfo;
            }
            if (aliasInfoForCurrentStmt != null) {
                for (Symbol definedSym : definedSymbolsInProc) {   // def symbols for the proc
                    Set<PointsToRel> aliasSetAffectingCurrentStmt = aliasInfoForCurrentStmt.get(definedSym);
                    if (aliasSetAffectingCurrentStmt != null) {
                        for (PointsToRel aliasInstance : aliasSetAffectingCurrentStmt) {
                            if (symForCurrentEx.getSymbolName().equals(aliasInstance.getPointedToSymbol().getSymbolName())) {
                                for (int i = 0; i < defTargetArray.length; i++) {
                                    Expression defID = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                                    if (defID.toString().equals(aliasInstance.getPointerSymbol().getSymbolName())) {
//                                        System.out.println("[" + proc.getSymbolName() + "]EQUAL: defID: " + defID + ", aliasPointer: " + aliasInstance.getPointerSymbol().getSymbolName() + ", currentUse: " + currentUse + ", stmt: " + nodeIR);
//                                        if (SymbolTools.isPointer(defTargetArray[i].getExpression()) == false) {
                                        if (ChainTools.checkInBitSet(cfgNode, i)) {
                                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                                        }
//                                        }
                                    }
                                }
                            } else if (symForCurrentEx.getSymbolName().equals(aliasInstance.getPointerSymbol().getSymbolName())) {
                                for (int i = 0; i < defTargetArray.length; i++) {
                                    Expression defID = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                                    if (defID.toString().equals(aliasInstance.getPointedToSymbol().getSymbolName())) {
//                                        System.out.println("[" + proc.getSymbolName() + "]EQUAL: defID: " + defID + ", aliasPointed: " + aliasInstance.getPointedToSymbol().getSymbolName() + ", currentUse: " + currentUse + ", stmt: " + nodeIR);
//                                        if (SymbolTools.isPointer(defTargetArray[i].getExpression()) == false) {
                                        if (ChainTools.checkInBitSet(cfgNode, i)) {
                                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                                        }
//                                        }
                                    }
                                }
                            } else {
                                // Do nothing
                            }
                        } // for (PointsToRel aliasInstance : aliasSetAffectingCurrentStmt)
                    } // if (aliasSetAffectingCurrentStmt != null)
                } // for (Symbol definedSym : definedSymbolsInProc)
                //
            } // if (aliasInfoForCurrentStmt != null)
        }
    }

    private void compute(Procedure proc) {
        AnalysisTarget[] defTargetArray = getDefTargetArray(proc);

        CFGraph cfg = cfgMap.get(proc);
        Iterator iter = cfg.iterator();
        while (iter.hasNext()) {
            DFANode cfgNode = (DFANode) iter.next();
            Object nodeIR = CFGraph.getIR(cfgNode);
            if (nodeIR instanceof VariableDeclarator) {
                if (ChainTools.isDefinedArrayDeclarator((VariableDeclarator) nodeIR)) {
                    generateChainForDefinedArrayDeclarator(cfgNode, defTargetArray, (VariableDeclarator) nodeIR, proc);
                } else {
                    generateChainForPlainDeclarator(cfgNode, defTargetArray, (VariableDeclarator) nodeIR, proc);
                }
            } else if (nodeIR instanceof NestedDeclarator) {
                if (ChainTools.isDefinedArrayNestedDeclarator((NestedDeclarator) nodeIR)) {
                    generateChainForDefinedArrayNestedDeclarator(cfgNode, defTargetArray, (NestedDeclarator) nodeIR, proc);
                } else {
                    generateChainForPlainNestedDeclarator(cfgNode, defTargetArray, (NestedDeclarator) nodeIR, proc);
                }
            } else if (nodeIR instanceof SwitchStatement) {
                SwitchStatement switchStmt = (SwitchStatement) nodeIR;
                Expression expInSwitch = switchStmt.getExpression();
//                System.out.println("##### SwitchStmt.getExpression: " + expInSwitch.toString());
//                System.out.println("################# SwitchStmt: " + switchStmt.toString());
                List<Expression> useList = ChainTools.getUseList(expInSwitch);
                for (Expression currentUse : useList) {
                    if (ChainTools.getIDExpression(currentUse) != null) {
                        if (ChainTools.isArrayAccess(currentUse)) {
                            generateChainForArrayAccess(currentUse, defTargetArray, cfgNode, proc);
                        } else if (ChainTools.isPointerAccess(currentUse)) {
                            generateChainForPointerAccess(currentUse, defTargetArray, cfgNode, proc);
                        } else if (ChainTools.isStructureAccess(currentUse, proc)) {
                            generateChainForStructure(currentUse, defTargetArray, cfgNode, proc);
                        } else {
                            generateChainForPlainExpression(currentUse, defTargetArray, cfgNode, proc);
                        }
                        generateChainForAlias(nodeIR, currentUse, defTargetArray, cfgNode, proc);
                    }
                } // for (Expression currentUse : useTargetSet)
            } else if (nodeIR instanceof Traversable) {
//                System.out.println("Switch Next: " + nodeIR.toString());
                Traversable traversableNodeIR = (Traversable) nodeIR;
                List<Expression> useList = ChainTools.getUseList(traversableNodeIR);
                for (Expression currentUse : useList) {
                    if (ChainTools.getIDExpression(currentUse) != null) {
                        if (ChainTools.isArrayAccess(currentUse)) {
                            generateChainForArrayAccess(currentUse, defTargetArray, cfgNode, proc);
                        } else if (ChainTools.isPointerAccess(currentUse)) {
                            if (ChainTools.isStructureAccess(currentUse, proc)) {
                                generateChainForStructure(currentUse, defTargetArray, cfgNode, proc);
                            } else {
                                generateChainForPointerAccess(currentUse, defTargetArray, cfgNode, proc);
                            }
                        } else if (ChainTools.isStructureAccess(currentUse, proc)) {
                            generateChainForStructure(currentUse, defTargetArray, cfgNode, proc);
                        } else {
                            generateChainForPlainExpression(currentUse, defTargetArray, cfgNode, proc);
                        }
                        generateChainForAlias(nodeIR, currentUse, defTargetArray, cfgNode, proc);
                    }
                } // for (Expression currentUse : useTargetSet)
                if (IRTools.containsFunctionCall(traversableNodeIR)) {
                    BitSet inBitSet = cfgNode.getData("InSet");
                    Set<DFANode> callNodeSetRef = cfgNode.getData("psg_call_ref");
                    if (callNodeSetRef != null) {
                        for (DFANode callNode : callNodeSetRef) {
                            Set<AnalysisTarget> inUSE = callNode.getData("INuse");
//                            Expression callArg = (Expression) callNode.getData("arg");
                            List<Expression> argsList = callNode.getData("arg");
                            for (Expression callArg: argsList) {
	                            if (callArg instanceof ConditionalExpression) {
	                                Expression trueEx = ((ConditionalExpression) callArg).getTrueExpression();
	                                if (trueEx instanceof BinaryExpression) {
	                                    List<Expression> defList = ChainTools.getUseList(trueEx);
	                                    for (Expression def : defList) {
	                                        Expression idEx = ChainTools.getIDExpression(def);
	                                        if (idEx != null) {
	                                            for (AnalysisTarget target : inUSE) {
	                                                for (int i = 0; i < defTargetArray.length; i++) {
	                                                    if (inBitSet.get(i)) {
	                                                        if (defTargetArray[i].getExpression().equals(idEx)) {
	                                                            defTargetArray[i].addUseChain(target);
	                                                            addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
	                                                        }
	                                                    }
	                                                }
	                                            }
	                                        }
	                                    }
	                                } else {
	                                    Expression idEx = ChainTools.getIDExpression(trueEx);
	                                    if (idEx != null) {
	                                        for (AnalysisTarget target : inUSE) {
	                                            for (int i = 0; i < defTargetArray.length; i++) {
	                                                if (inBitSet.get(i)) {
	                                                    if (defTargetArray[i].getExpression().equals(idEx)) {
	                                                        defTargetArray[i].addUseChain(target);
	                                                        addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
	                                                    }
	                                                }
	                                            }
	                                        }
	                                    }
	                                }
	                                Expression falseEx = ((ConditionalExpression) callArg).getTrueExpression();
	                                if (falseEx instanceof BinaryExpression) {
	                                    List<Expression> defList = ChainTools.getUseList(falseEx);
	                                    for (Expression def : defList) {
	                                        Expression idEx = ChainTools.getIDExpression(def);
	                                        if (idEx != null) {
	                                            for (AnalysisTarget target : inUSE) {
	                                                for (int i = 0; i < defTargetArray.length; i++) {
	                                                    if (inBitSet.get(i)) {
	                                                        if (defTargetArray[i].getExpression().equals(idEx)) {
	                                                            defTargetArray[i].addUseChain(target);
	                                                            addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
	                                                        }
	                                                    }
	                                                }
	                                            }
	                                        }
	                                    }
	                                } else {
	                                    Expression idEx = ChainTools.getIDExpression(falseEx);
	                                    if (idEx != null) {
	                                        for (AnalysisTarget target : inUSE) {
	                                            for (int i = 0; i < defTargetArray.length; i++) {
	                                                if (inBitSet.get(i)) {
	                                                    if (defTargetArray[i].getExpression().equals(idEx)) {
	                                                        defTargetArray[i].addUseChain(target);
	                                                        addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
	                                                    }
	                                                }
	                                            }
	                                        }
	                                    }
	                                }
	                            } else if (callArg instanceof BinaryExpression) {
	                                List<Expression> defList = ChainTools.getUseList(callArg);
	                                for (Expression def : defList) {
	                                    Expression idEx = ChainTools.getIDExpression(def);
	                                    if (idEx != null) {
	                                        for (AnalysisTarget target : inUSE) {
	                                            for (int i = 0; i < defTargetArray.length; i++) {
	                                                if (inBitSet.get(i)) {
	                                                    if (defTargetArray[i].getExpression().equals(idEx)) {
	                                                        defTargetArray[i].addUseChain(target);
	                                                        addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
	                                                    }
	                                                }
	                                            }
	                                        }
	                                    }
	                                }
	                            } else {
	                                Expression idEx = ChainTools.getIDExpression(callArg);
	                                if (idEx != null) {
	                                    for (AnalysisTarget target : inUSE) {
	                                        for (int i = 0; i < defTargetArray.length; i++) {
	                                            if (inBitSet.get(i)) {
	                                                if (defTargetArray[i].getExpression().equals(idEx)) {
	                                                    defTargetArray[i].addUseChain(target);
	                                                    addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
	                                                }
	                                            }
	                                        }
	                                    }
	                                }
	                            }
                            }
                        }
                    }
                    Set<DFANode> callNodeSetGlobal = cfgNode.getData("psg_call_global");
                    if (callNodeSetGlobal != null) {
                        AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseSet.size()];
                        globalUseSet.toArray(globalUseArray);
                        for (DFANode callNode : callNodeSetGlobal) {
                            BitSet bitInUSE = callNode.getData("INuse");
                            for (int i = 0; i < defTargetArray.length; i++) {
                                if (inBitSet.get(i)) {
                                    for (int idx = 0; idx < globalUseArray.length; idx++) {
                                        if (bitInUSE.get(idx)) {
                                            if (defTargetArray[i].getExpression().equals(globalUseArray[idx].getExpression())) {
                                                defTargetArray[i].addUseChain(globalUseArray[idx]);
                                                addUseDefChain(defTargetArray[i], globalUseArray[idx].getExpression(), globalUseArray[idx].getDFANode(), globalUseArray[idx].getProcedure());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Set<DFANode> returnNodeSetGlobal = cfgNode.getData("psg_return_global");
                    if (returnNodeSetGlobal != null) {
                        AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseSet.size()];
                        globalUseSet.toArray(globalUseArray);
                        for (DFANode returnNode : returnNodeSetGlobal) {
                            BitSet bitInUSE = returnNode.getData("INuse");
                            for (int i = 0; i < defTargetArray.length; i++) {
                                if (inBitSet.get(i)) {
                                    for (int idx = 0; idx < globalUseArray.length; idx++) {
                                        if (bitInUSE.get(idx)) {
                                            if (defTargetArray[i].getExpression().equals(globalUseArray[idx].getExpression())) {
                                                defTargetArray[i].addUseChain(globalUseArray[idx]);
                                                addUseDefChain(defTargetArray[i], globalUseArray[idx].getExpression(), globalUseArray[idx].getDFANode(), globalUseArray[idx].getProcedure());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // at exit node, should handle INuse
            if (ChainTools.isExitNode(cfgNode, cfg)) {
                BitSet inBitSet = cfgNode.getData("InSet");
                // Ref
                Set<DFANode> exitNodeSetRef = cfgNode.getData("psg_exit_ref");
                if (exitNodeSetRef != null) {
                    for (DFANode exitNode : exitNodeSetRef) {
                        Set<AnalysisTarget> inUSE = exitNode.getData("INuse");
                        IDExpression param = ChainTools.getIDExpression(((AnalysisTarget) exitNode.getData("param")).getExpression());
                        for (AnalysisTarget target : inUSE) {
                            for (int i = 0; i < defTargetArray.length; i++) {
                                if (inBitSet.get(i)) {
                                    if (defTargetArray[i].getExpression().equals(param)) {
                                        defTargetArray[i].addUseChain(target);
                                        addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                    }
                                }
                            }
                        }
                    }
                }
                // Global
                DFANode exitNodeGlobal = cfgNode.getData("psg_exit_global");
                if (exitNodeGlobal != null) {
                    AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseSet.size()];
                    globalUseSet.toArray(globalUseArray);
                    BitSet bitInUSE = exitNodeGlobal.getData("INuse");
                    for (int i = 0; i < defTargetArray.length; i++) {
                        if (inBitSet.get(i)) {
                            for (int idx = 0; idx < globalUseArray.length; idx++) {
                                if (bitInUSE.get(idx)) {
                                    if (defTargetArray[i].getExpression().equals(globalUseArray[idx].getExpression())) {
                                        defTargetArray[i].addUseChain(globalUseArray[idx]);
                                        addUseDefChain(defTargetArray[i], globalUseArray[idx].getExpression(), globalUseArray[idx].getDFANode(), globalUseArray[idx].getProcedure());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public void printDefUseChain() {
        if (isAnalysisDone == false) {
            this.start();
        }
        for (Procedure proc : procList) {
            System.out.println("########################## (printDefUseChain) Procedure: " + proc.getSymbolName());
            AnalysisTarget[] defTargetArray = getDefTargetArray(proc);
            for (int i = 0; i < defTargetArray.length; i++) {
                System.out.println("Def[" + i + "]: " + defTargetArray[i].getExpression() + ", IR: " + defTargetArray[i].getDFANode().getData("ir"));
                List<Traversable> useSet = getUseList(defTargetArray[i].getExpression());
                for (Traversable useNode : useSet) {
                    Procedure useProc = ChainTools.getParentProcedure(useNode, program);
                    if (useProc == null) {
                        System.out.println("  --> Use: " + useNode + ", proc: null");
                    } else {
                        System.out.println("  --> Use: " + useNode + ", proc: " + useProc.getSymbolName());
                    }
                }
            }
            System.out.println("#######################################################################");
        }
    }

    public void printUseDefChain() {
        if (isAnalysisDone == false) {
            this.start();
        }
        for (Procedure proc : procList) {
            System.out.println("########################## (printUseDefChain) Procedure: " + proc.getSymbolName());
            AnalysisTarget[] useTargetArray = getUseTargetArray(proc);
            if (useTargetArray != null) {
                for (int i = 0; i < useTargetArray.length; i++) {
                    System.out.println("Use[" + i + "]: " + useTargetArray[i].getExpression() + ", IR: " + useTargetArray[i].getDFANode().getData("ir"));
                    List<Traversable> defSet = getDefList(useTargetArray[i].getExpression());
                    for (Traversable defNode : defSet) {
                        Procedure defProc = ChainTools.getParentProcedure(defNode, program);
                        if (defProc == null) {
                            System.out.println("  --> Def: " + defNode + ", proc: null");
                        } else {
                            System.out.println("  --> Def: " + defNode + ", proc: " + defProc.getSymbolName());
                        }
                    }
                }
            }
            System.out.println("#######################################################################");
        }
    }

    public List<String> getUDChainForTesting(Procedure proc) {
        if (isAnalysisDone == false) {
            this.start();
        }
        List result = new ArrayList<String>();
        CFGraph cfg = cfgMap.get(proc);
        Iterator iter = cfg.iterator();
        while (iter.hasNext()) {
            DFANode dfaNode = (DFANode) iter.next();
            Object o = CFGraph.getIR(dfaNode);
            if (o == null) {
                continue;
            }
            if (o instanceof VariableDeclarator) {
                List<Expression> useSet = ChainTools.getUseListInDec((VariableDeclarator) o);
                for (Expression myEx : useSet) {
                    result.add(myEx.toString());
                    List<Traversable> trDefSet = getDefList(myEx);
                    for (Traversable stmt : trDefSet) {
                        result.add(stmt.toString());
                    }
                }
            } else if (o instanceof Traversable) {
                Traversable tr = (Traversable) o;
                List<Expression> useList = ChainTools.getUseList(tr);
                for (Expression myEx : useList) {
                    if (ChainTools.getIDExpression(myEx) != null) {
                        result.add(myEx.toString());
                        List<Traversable> trDefSet = getDefList(myEx);
                        for (Traversable stmt : trDefSet) {
                            result.add(stmt.toString());
                        }
                    }
                }
            } else {
                throw new RuntimeException(o.toString() + ", class: " + o.getClass().getCanonicalName());
            }
        }
        return result;
    }

    public Map<Procedure, CFGraph> getCfgMap() {
        return cfgMap;
    }

    // Implementation of DefUseChain interface
    public List<Traversable> getUseList(Expression def) {
        if (def == null) {
            return null;
        }
        List returnUseSet = new ArrayList();
        Procedure proc = ChainTools.getParentProcedure(def, program);
        if (proc != null) {
            Set<AnalysisTarget> defTargetSet = defTargetSetMap.get(proc);
            for (AnalysisTarget defTarget : defTargetSet) {
                if (System.identityHashCode(def) == System.identityHashCode(defTarget.getExpression())) {
                    Set<AnalysisTarget> useChain = defTarget.getUseChain();
                    if (useChain != null) {
                        for (AnalysisTarget useTarget : useChain) {
                            returnUseSet.add(useTarget.getDFANode().getData("ir"));
                        }
                    }
                }
            }
            return returnUseSet;
        } else {
            throw new RuntimeException("UDChainTools.getParentProcedure returns null for " + def);
        }
    }

    public List<Traversable> getLocalUseList(Expression def) {
        if (def == null) {
            return null;
        }
        List returnUseSet = new ArrayList();
        Procedure proc = ChainTools.getParentProcedure(def, program);
        if (proc != null) {
            Set<AnalysisTarget> defTargetSet = defTargetSetMap.get(proc);
            for (AnalysisTarget defTarget : defTargetSet) {
                if (System.identityHashCode(def) == System.identityHashCode(defTarget.getExpression())) {
                    Set<AnalysisTarget> useChain = defTarget.getUseChain();
                    if (useChain != null) {
                        for (AnalysisTarget use : useChain) {
                            if (proc.equals(use.getProcedure())) {
                                returnUseSet.add(use.getDFANode().getData("ir"));
                            }
                        }
                    }
                }
            }
            return returnUseSet;
        } else {
            throw new RuntimeException("UDChainTools.getParentProcedure returns null for " + def);
        }
    }

    public boolean isReachable(Expression def, Expression use) {
        if (def == null || use == null) {
            return false;
        }
        Procedure proc = ChainTools.getParentProcedure(def, program);
        if (proc != null) {
            Set<AnalysisTarget> defTargetSet = defTargetSetMap.get(proc);
            for (AnalysisTarget defTarget : defTargetSet) {
                if (System.identityHashCode(def) == System.identityHashCode(defTarget.getExpression())) {
                    Set<AnalysisTarget> useChain = defTarget.getUseChain();
                    if (useChain != null) {
                        for (AnalysisTarget useTarget : useChain) {
                            if (System.identityHashCode(use) == System.identityHashCode(useTarget.getExpression())) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    // Implementation of UseDefChain interface
    public List<Traversable> getDefList(Expression use) {
        if (use == null) {
            return null;
        }
        List returnDefSet = new ArrayList();
        Procedure proc = ChainTools.getParentProcedure(use, program);
        if (proc != null) {
            Set<AnalysisTarget> useTargetSet = useTargetSetMap.get(proc);
            for (AnalysisTarget useTarget : useTargetSet) {
                if (System.identityHashCode(use) == System.identityHashCode(useTarget.getExpression())) {
                    Set<AnalysisTarget> defChain = useTarget.getDefChain();
                    if (defChain != null) {
                        for (AnalysisTarget def : defChain) {
                            returnDefSet.add(def.getDFANode().getData("ir"));
                        }
                    }
                }
            }
            return returnDefSet;
        } else {
            throw new RuntimeException("UDChainTools.getParentProcedure returns null for " + use);
        }
    }
    
    public Set<DFANode> getDefDFANodeSet(Expression use) {
        if (use == null) {
            return null;
        }
        Set returnDefSet = new HashSet();
        Procedure proc = ChainTools.getParentProcedure(use, program);
        if (proc != null) {
            Set<AnalysisTarget> useTargetSet = useTargetSetMap.get(proc);
            for (AnalysisTarget useTarget : useTargetSet) {
                if (System.identityHashCode(use) == System.identityHashCode(useTarget.getExpression())) {
                    Set<AnalysisTarget> defChain = useTarget.getDefChain();
                    if (defChain != null) {
                        for (AnalysisTarget def : defChain) {
                            returnDefSet.add(def.getDFANode());
                        }
                    }
                }
            }
            return returnDefSet;
        } else {
            throw new RuntimeException("UDChainTools.getParentProcedure returns null for " + use);
        }
    }

    public List<Traversable> getLocalDefList(Expression use) {
        if (use == null) {
            return null;
        }
        List returnDefSet = new ArrayList();
        Procedure proc = ChainTools.getParentProcedure(use, program);
        if (proc != null) {
            Set<AnalysisTarget> useList = useTargetSetMap.get(proc);
            for (AnalysisTarget useItem : useList) {
                if (System.identityHashCode(use) == System.identityHashCode(useItem.getExpression())) {
                    Set<AnalysisTarget> defChain = useItem.getDefChain();
                    if (defChain != null) {
                        for (AnalysisTarget defTarget : defChain) {
                            if (proc.equals(defTarget.getProcedure())) {
                                returnDefSet.add(defTarget.getDFANode().getData("ir"));
                            }
                        }
                    }
                }
            }
            return returnDefSet;
        } else {
            throw new RuntimeException("UDChainTools.getParentProcedure returns null for " + use);
        }
    }
}
